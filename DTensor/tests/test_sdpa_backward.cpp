/**
 * test_sdpa_backward.cpp
 *
 * Standalone test to verify that a manual SDPA backward produces the same
 * gradients as the autograd-based sdpa_backward_op. This is used to debug
 * the Di correction needed for context parallel ring attention.
 *
 * Build: make test_sdpa_backward
 * Run:   ./test_sdpa_backward_exec
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "ops/Kernels.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/FusedKernels.cuh"

// Include the SDPA ops we want to test
#include "gpt2_cp_test/context_parallel/SDPAOp.h"

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// Manual SDPA backward (candidate implementation to verify)
// ---------------------------------------------------------------------------
std::vector<Tensor> sdpa_backward_manual_test(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& grad_output,
    bool is_causal,
    float scale,
    const Tensor& Di)
{
    // Recompute forward with raw ops
    Tensor q_scaled = q * scale;
    Tensor k_t = k.transpose(-2, -1);
    Tensor scores = OwnTensor::matmul(q_scaled, k_t);

    Tensor P;
    if (is_causal) {
        P = OwnTensor::fused_tril_softmax(scores, 0);
    } else {
        int64_t last_dim = scores.ndim() - 1;
        Tensor max_s = reduce_max(scores, {last_dim}, true);
        Tensor exp_s = OwnTensor::exp(scores - max_s);
        Tensor sum_s = reduce_sum(exp_s, {last_dim}, true);
        P = exp_s / sum_s;
    }

    // dV = P^T @ grad_output
    Tensor P_t = P.transpose(-2, -1);
    Tensor grad_v = OwnTensor::matmul(P_t, grad_output);

    // dP = grad_output @ V^T
    Tensor v_t = v.transpose(-2, -1);
    Tensor dP = OwnTensor::matmul(grad_output, v_t);

    // dS = P * (dP - Di)
    Tensor dS = P * (dP - Di);

    // dK = dS^T @ q_scaled
    Tensor dS_t = dS.transpose(-2, -1);
    Tensor grad_k = OwnTensor::matmul(dS_t, q_scaled);

    // dQ = (dS @ K) * scale
    Tensor grad_q = OwnTensor::matmul(dS, k) * scale;

    return {grad_q, grad_k, grad_v};
}

// ---------------------------------------------------------------------------
// Utility: compare two tensors element-wise
// ---------------------------------------------------------------------------
struct CompareResult {
    float max_abs_diff;
    float max_rel_diff;
    float mean_abs_diff;
    int64_t num_elements;
    int64_t num_mismatches;  // abs diff > tolerance
};

CompareResult compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-5f) {
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();

    float* a_data = a_cpu.data<float>();
    float* b_data = b_cpu.data<float>();
    int64_t n = a_cpu.numel();

    CompareResult res = {0, 0, 0, n, 0};
    double sum_diff = 0;

    for (int64_t i = 0; i < n; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float denom = std::max(std::abs(a_data[i]), std::abs(b_data[i]));
        float rel = (denom > 1e-8f) ? diff / denom : 0.0f;

        if (diff > res.max_abs_diff) res.max_abs_diff = diff;
        if (rel > res.max_rel_diff) res.max_rel_diff = rel;
        sum_diff += diff;
        if (diff > tol) res.num_mismatches++;
    }
    res.mean_abs_diff = static_cast<float>(sum_diff / n);
    return res;
}

void print_compare(const char* name, const CompareResult& r) {
    std::cout << "  " << std::setw(8) << name
              << " | max_abs: " << std::scientific << std::setprecision(4) << r.max_abs_diff
              << " | max_rel: " << r.max_rel_diff
              << " | mean_abs: " << r.mean_abs_diff
              << " | mismatches(>1e-5): " << r.num_mismatches << "/" << r.num_elements
              << std::endl;
}

// ---------------------------------------------------------------------------
// Print first/last few elements of a tensor for visual inspection
// ---------------------------------------------------------------------------
void print_tensor_snippet(const char* name, const Tensor& t, int n = 5) {
    Tensor cpu = t.to_cpu();
    float* data = cpu.data<float>();
    int64_t total = cpu.numel();

    std::cout << "  " << name << " [" << total << " elems]: [";
    for (int i = 0; i < std::min((int64_t)n, total); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << data[i];
    }
    std::cout << " ... ";
    for (int64_t i = std::max((int64_t)0, total - n); i < total; ++i) {
        std::cout << data[i];
        if (i < total - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// ---------------------------------------------------------------------------
// Test: compare autograd backward vs manual backward
// ---------------------------------------------------------------------------
void test_sdpa_backward_comparison(bool is_causal, int B, int H, int T, int D) {
    std::cout << "\n=== Test: is_causal=" << (is_causal ? "true" : "false")
              << " B=" << B << " H=" << H << " T=" << T << " D=" << D
              << " ===" << std::endl;

    DeviceIndex device(Device::CUDA, 0);
    cudaSetDevice(0);

    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                        .with_device(device)
                                        .with_req_grad(true);
    TensorOptions opts_nograd = TensorOptions().with_dtype(Dtype::Float32)
                                               .with_device(device);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Create random Q, K, V
    Tensor q = Tensor::randn<float>(Shape({{B, H, T, D}}), opts, 42, 0.02f);
    Tensor k = Tensor::randn<float>(Shape({{B, H, T, D}}), opts, 43, 0.02f);
    Tensor v = Tensor::randn<float>(Shape({{B, H, T, D}}), opts, 44, 0.02f);

    // Create random grad_output
    Tensor grad_out = Tensor::randn<float>(Shape({{B, H, T, D}}), opts_nograd, 45, 0.01f);

    // ----- Autograd backward -----
    Tensor q1 = q.clone(); q1.set_requires_grad(true);
    Tensor k1 = k.clone(); k1.set_requires_grad(true);
    Tensor v1 = v.clone(); v1.set_requires_grad(true);
    Tensor g1 = grad_out.clone();

    std::vector<Tensor> auto_grads = sdpa_backward_op(q1, k1, v1, g1, is_causal, scale);
    Tensor auto_dq = auto_grads[0];
    Tensor auto_dk = auto_grads[1];
    Tensor auto_dv = auto_grads[2];

    // ----- Manual backward with Di = rowsum(grad_out * out) -----
    // First compute out for Di
    Tensor q_det = q.detach();
    Tensor k_det = k.detach();
    Tensor v_det = v.detach();

    // Recompute forward to get out for Di
    Tensor q_sc = q_det * scale;
    Tensor k_t_det = k_det.transpose(-2, -1);
    Tensor scores_det = OwnTensor::matmul(q_sc, k_t_det);

    Tensor P_det;
    if (is_causal) {
        P_det = OwnTensor::fused_tril_softmax(scores_det, 0);
    } else {
        int64_t ld = scores_det.ndim() - 1;
        Tensor mx = reduce_max(scores_det, {ld}, true);
        Tensor ex = OwnTensor::exp(scores_det - mx);
        Tensor sm = reduce_sum(ex, {ld}, true);
        P_det = ex / sm;
    }
    Tensor out_det = OwnTensor::matmul(P_det, v_det);

    // Di = rowsum(grad_out * out)  -- this is what autograd computes internally
    Tensor Di_local = reduce_sum(grad_out * out_det, {-1}, true);

    std::vector<Tensor> manual_grads = sdpa_backward_manual_test(
        q_det, k_det, v_det, grad_out, is_causal, scale, Di_local);

    Tensor man_dq = manual_grads[0];
    Tensor man_dk = manual_grads[1];
    Tensor man_dv = manual_grads[2];

    // ----- Compare -----
    std::cout << "Autograd vs Manual (with Di_local = rowsum(grad*out)):" << std::endl;
    print_compare("dQ", compare_tensors(auto_dq, man_dq));
    print_compare("dK", compare_tensors(auto_dk, man_dk));
    print_compare("dV", compare_tensors(auto_dv, man_dv));

    // Print snippets for visual inspection
    std::cout << "\nAutograd dQ:" << std::endl;
    print_tensor_snippet("auto_dq", auto_dq);
    std::cout << "Manual dQ:" << std::endl;
    print_tensor_snippet("man_dq", man_dq);

    std::cout << "\nAutograd dK:" << std::endl;
    print_tensor_snippet("auto_dk", auto_dk);
    std::cout << "Manual dK:" << std::endl;
    print_tensor_snippet("man_dk", man_dk);

    std::cout << "\nAutograd dV:" << std::endl;
    print_tensor_snippet("auto_dv", auto_dv);
    std::cout << "Manual dV:" << std::endl;
    print_tensor_snippet("man_dv", man_dv);

    // ----- Test with a DIFFERENT Di (simulating D_global) -----
    // Use Di_global = Di_local * 0.7 (arbitrary different value)
    std::cout << "\n--- With modified Di (Di * 0.7, simulating D_global correction) ---"
              << std::endl;

    Tensor Di_modified = Di_local * 0.7f;
    std::vector<Tensor> manual_grads_mod = sdpa_backward_manual_test(
        q_det, k_det, v_det, grad_out, is_causal, scale, Di_modified);

    std::cout << "Autograd vs Manual (Di_modified):" << std::endl;
    print_compare("dQ", compare_tensors(auto_dq, manual_grads_mod[0]));
    print_compare("dK", compare_tensors(auto_dk, manual_grads_mod[1]));
    print_compare("dV", compare_tensors(auto_dv, manual_grads_mod[2]));

    std::cout << "dV should be IDENTICAL (Di doesn't affect dV):" << std::endl;
    print_compare("dV", compare_tensors(man_dv, manual_grads_mod[2]));
}


int main() {
    std::cout << "=== SDPA Backward Comparison Test ===" << std::endl;

    // Test non-causal
    test_sdpa_backward_comparison(false, 2, 1, 8, 16);

    // Test causal
    test_sdpa_backward_comparison(true, 2, 1, 8, 16);

    // Test with larger dimensions (closer to real usage)
    test_sdpa_backward_comparison(true, 1, 1, 64, 32);

    std::cout << "\n=== All tests complete ===" << std::endl;
    return 0;
}