// Tensor library sanity test for the narrow_view + clone + cat pattern
// used in ContextParallelBackward.h's LB-specific accumulation paths.
//
// Tests two patterns against hand-built expected results:
//   A) "partial dQ" pattern: cat({head.clone(), tail.clone() + add}, dim)
//   B) "head-half pad-then-add" pattern: original + cat({add_half, zeros}, dim)
//
// If either diverges from the expected result, the Tensor library
// (make_shards_inplace_axis / narrow_view / clone / cat) has a bug that
// would explain the LB-specific accumulation discrepancy observed in
// cp_bwd_isolated_test.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "TensorLib.h"

using namespace OwnTensor;

namespace {

constexpr int B = 1, H = 2, T = 64, D = 64;
constexpr int half_T = T / 2;
constexpr int seq_dim = 2;

float max_abs_diff(const Tensor &a, const Tensor &b) {
  Tensor ah = a.to_cpu();
  Tensor bh = b.to_cpu();
  const float *ap = ah.data<float>();
  const float *bp = bh.data<float>();
  int64_t n = a.numel();
  if (b.numel() != n) {
    std::cerr << "shape mismatch: " << n << " vs " << b.numel() << "\n";
    return -1.0f;
  }
  float m = 0;
  for (int64_t i = 0; i < n; ++i) {
    float d = std::abs(ap[i] - bp[i]);
    if (d > m) m = d;
  }
  return m;
}

// Build a tensor with sequential float values [0, 1, 2, ..., numel-1]
Tensor arange_tensor(const Shape &shape, DeviceIndex device) {
  TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
  Tensor h = Tensor::empty(shape, cpu_opts);
  int64_t n = h.numel();
  float *p = h.data<float>();
  for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);
  return h.to(device);
}

}  // namespace

int main(int argc, char **argv) {
  cudaSetDevice(0);
  DeviceIndex device(Device::CUDA, 0);

  // ===== Test A: partial dQ pattern =====
  // C++ code under test (from CPB.h):
  //   gq_1st = grad_q.narrow_view(2, 0, half_T);
  //   gq_2nd = grad_q.narrow_view(2, half_T, half_T);
  //   grad_q = cat({gq_1st.clone(), gq_2nd.clone() + grad_q_step}, 2);
  //
  // Inputs:
  //   grad_q  = arange [1,2,T,D]
  //   grad_q_step = arange [1,2,T/2,D] + 1000  (so values are distinct from grad_q)
  //
  // Expected:
  //   head = grad_q[:, :, 0:T/2, :]  (unchanged)
  //   tail = grad_q[:, :, T/2:T, :] + grad_q_step
  //   result = cat(head, tail, dim=2)
  std::cout << "=== Test A: partial dQ pattern ===\n";
  {
    Shape full_shape({{B, H, T, D}});
    Shape half_shape({{B, H, half_T, D}});

    Tensor grad_q = arange_tensor(full_shape, device);
    Tensor grad_q_step_host = arange_tensor(half_shape, DeviceIndex(Device::CPU, 0));
    // Add 1000 to each element of grad_q_step on host, then move to device
    {
      float *p = grad_q_step_host.data<float>();
      for (int64_t i = 0; i < grad_q_step_host.numel(); ++i) p[i] += 1000.0f;
    }
    Tensor grad_q_step = grad_q_step_host.to(device);

    // ----- under test -----
    Tensor gq_1st = grad_q.narrow_view(seq_dim, 0, half_T);
    Tensor gq_2nd = grad_q.narrow_view(seq_dim, half_T, half_T);
    Tensor result = Tensor::cat({gq_1st.clone(), gq_2nd.clone() + grad_q_step}, seq_dim);

    // ----- expected (hand-built CPU reference) -----
    TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor expected_cpu = Tensor::empty(full_shape, cpu_opts);
    float *e = expected_cpu.data<float>();
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
          for (int64_t d = 0; d < D; ++d) {
            int64_t idx = ((b * H + h) * T + t) * D + d;
            float v = static_cast<float>(idx);
            if (t >= half_T) {
              // tail half: original + grad_q_step
              int64_t t_local = t - half_T;
              int64_t step_idx = ((b * H + h) * half_T + t_local) * D + d;
              v += static_cast<float>(step_idx) + 1000.0f;
            }
            e[idx] = v;
          }
        }
      }
    }
    Tensor expected = expected_cpu.to(device);

    float diff = max_abs_diff(result, expected);
    std::cout << "  max_abs_diff = " << diff << (diff == 0.0f ? " ✓" : " ✗") << "\n";
  }

  // ===== Test A2: full sequence mirroring actual CPB.h backward =====
  //
  // step 0:  grad_q = zeros + step0_dQ                (line 239 in CPB.h)
  // step 1:  grad_q = cat({head.clone(), tail.clone() + step1_dQ}, 2)  (line 232-237)
  //
  // This mirrors the actual call sequence — grad_q at step 1 is the result
  // of an add (not a fresh arange), so its storage may have different
  // characteristics that could affect narrow_view + clone.
  std::cout << "=== Test A2: full step-0 add + step-1 partial update ===\n";
  {
    Shape full_shape({{B, H, T, D}});
    Shape half_shape({{B, H, half_T, D}});

    // step 0 inputs
    Tensor step0_dQ = arange_tensor(full_shape, device);
    Tensor zeros_full = Tensor::zeros(full_shape, step0_dQ.opts());

    // step 1 input (offset by 1000 to distinguish)
    Tensor step1_dQ_host = arange_tensor(half_shape, DeviceIndex(Device::CPU, 0));
    {
      float *p = step1_dQ_host.data<float>();
      for (int64_t i = 0; i < step1_dQ_host.numel(); ++i) p[i] += 1000.0f;
    }
    Tensor step1_dQ = step1_dQ_host.to(device);

    // ----- under test: replicate CPB.h backward sequence -----
    Tensor grad_q = zeros_full + step0_dQ;  // mirrors `grad_q = grad_q + grad_q_step` at step 0
    Tensor gq_1st = grad_q.narrow_view(seq_dim, 0, half_T);
    Tensor gq_2nd = grad_q.narrow_view(seq_dim, half_T, half_T);
    Tensor result = Tensor::cat({gq_1st.clone(), gq_2nd.clone() + step1_dQ}, seq_dim);

    // ----- expected -----
    TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor expected_cpu = Tensor::empty(full_shape, cpu_opts);
    float *e = expected_cpu.data<float>();
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
          for (int64_t d = 0; d < D; ++d) {
            int64_t idx = ((b * H + h) * T + t) * D + d;
            float v = static_cast<float>(idx);  // step 0 contribution
            if (t >= half_T) {
              int64_t t_local = t - half_T;
              int64_t step_idx = ((b * H + h) * half_T + t_local) * D + d;
              v += static_cast<float>(step_idx) + 1000.0f;
            }
            e[idx] = v;
          }
        }
      }
    }
    Tensor expected = expected_cpu.to(device);

    float diff = max_abs_diff(result, expected);
    std::cout << "  max_abs_diff = " << diff << (diff == 0.0f ? " ✓" : " ✗") << "\n";

    // Also: check that the head half is preserved bit-for-bit through cat
    Tensor head_only = result.narrow_view(seq_dim, 0, half_T).clone();
    Tensor head_expected = grad_q.narrow_view(seq_dim, 0, half_T).clone();
    float head_diff = max_abs_diff(head_only, head_expected);
    std::cout << "  head_preserved_diff = " << head_diff
              << (head_diff == 0.0f ? " ✓" : " ✗") << "\n";
  }

  // ===== Test B: head-half pad-then-add pattern =====
  // C++ code under test (from CPB.h):
  //   zeros_half = Tensor::zeros(grad_k_step.shape(), grad_k_step.opts());
  //   gk_padded = cat({grad_k_step, zeros_half}, seq_dim);  // shape [B,H,T,D]
  //   grad_key = grad_key + gk_padded;
  //
  // Inputs:
  //   grad_key   = arange [1,2,T,D]
  //   grad_k_step = arange [1,2,T/2,D] + 1000
  //
  // Expected:
  //   head = grad_key[head] + grad_k_step
  //   tail = grad_key[tail]  (unchanged)
  std::cout << "=== Test B: head-half pad-then-add pattern ===\n";
  {
    Shape full_shape({{B, H, T, D}});
    Shape half_shape({{B, H, half_T, D}});

    Tensor grad_key = arange_tensor(full_shape, device);
    Tensor grad_k_step_host = arange_tensor(half_shape, DeviceIndex(Device::CPU, 0));
    {
      float *p = grad_k_step_host.data<float>();
      for (int64_t i = 0; i < grad_k_step_host.numel(); ++i) p[i] += 1000.0f;
    }
    Tensor grad_k_step = grad_k_step_host.to(device);
    Tensor zeros_half = Tensor::zeros(grad_k_step.shape(), grad_k_step.opts());

    // ----- under test -----
    Tensor gk_padded = Tensor::cat({grad_k_step, zeros_half}, seq_dim);
    Tensor result = grad_key + gk_padded;

    // ----- expected -----
    TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor expected_cpu = Tensor::empty(full_shape, cpu_opts);
    float *e = expected_cpu.data<float>();
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
          for (int64_t d = 0; d < D; ++d) {
            int64_t idx = ((b * H + h) * T + t) * D + d;
            float v = static_cast<float>(idx);
            if (t < half_T) {
              int64_t step_idx = ((b * H + h) * half_T + t) * D + d;
              v += static_cast<float>(step_idx) + 1000.0f;
            }
            e[idx] = v;
          }
        }
      }
    }
    Tensor expected = expected_cpu.to(device);

    float diff = max_abs_diff(result, expected);
    std::cout << "  max_abs_diff = " << diff << (diff == 0.0f ? " ✓" : " ✗") << "\n";
  }

  return 0;
}
