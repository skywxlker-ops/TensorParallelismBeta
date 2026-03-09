# Phase 1: Quick Wins & Correctness

## 1. Removed `output.wait()` in DEmbedding::forward

**File:** `CustomDNN/CustomDNN.cpp` (line ~371)

**What:** Removed the blocking `output.wait()` call after assigning the local embedding output.

**Why:** `DEmbedding` uses replicated weights — every rank holds the full embedding table and computes the full output independently. There is no pending collective (AllReduce/AllGather) on this tensor, so `wait()` was forcing a needless CPU-GPU synchronization barrier on every forward pass, stalling the pipeline.

**How:** Deleted the `output.wait()` line. The DTensor is returned directly after assigning the local tensor.

---

## 2. Cached `cudaGetDeviceCount` in DEmbedding::forward

**File:** `CustomDNN/CustomDNN.cpp` (line ~360)

**What:** Replaced the per-call `cudaGetDeviceCount(&num_devices)` with a `static` variable initialized once via a lambda.

**Why:** `cudaGetDeviceCount` is a CUDA runtime query that, while cheap, still involves a driver roundtrip. The number of GPUs on a node never changes during a process's lifetime, so querying it on every forward call is wasteful — especially when this runs thousands of times per training run.

**How:** Changed to `static int num_devices = [] { int n; cudaGetDeviceCount(&n); return n; }();` which evaluates exactly once (thread-safe in C++11+) and reuses the result for all subsequent calls.

---

## 3. Cached `scale_t` as a DAttention class member

**Files:**
- `CustomDNN/CustomDNN.h` (line ~330) — added member
- `CustomDNN/CustomDNN.cpp` (line ~289) — initialized in constructor
- `CustomDNN/gpt2_tests/gpt2_tp_test.cpp` (line ~129) — used in forward

**What:** Moved the attention scaling tensor (`1/sqrt(head_dim)`) from a per-call allocation to a pre-allocated class member `cached_scale_t_`.

**Why:** Every forward call was creating a new `Tensor::full` on the GPU — allocating device memory, launching a fill kernel, and constructing the tensor object. Since `head_dim` is constant for the lifetime of the model, this tensor never changes. Allocating it once in the constructor eliminates repeated GPU memory allocations and kernel launches from the hot path.

**How:**
1. Added `OwnTensor::Tensor cached_scale_t_` member to the `DAttention` class declaration in the header.
2. Initialized it at the end of the `DAttention` constructor with `Tensor::full(Shape{{1}}, ..., 1.0f / sqrt(head_dim))`.
3. Replaced the local `scale_t` variable in the forward function with `cached_scale_t_`.

---

# Phase 2: Communication Optimizations

## 4. Batched replicated grad AllReduce

**File:** `CustomDNN/gpt2_tests/gpt2_tp_test.cpp` (line ~665)

**What:** Replaced N individual AllReduce calls (one per replicated parameter gradient) with a single AllReduce on a packed flat buffer.

**Why:** Each AllReduce has fixed overhead — NCCL kernel launch, ring/tree setup, synchronization. With N replicated parameters (LayerNorm weights/biases, embeddings), that's N round-trips. Packing all grads into one contiguous buffer and doing a single AllReduce amortizes the fixed overhead, reducing total communication latency.

**How:**
1. First pass: collect all replicated grad pointers and sizes, compute total element count.
2. `cudaMalloc` a flat buffer of `total_numel` floats (once per step — negligible cost).
3. `cudaMemcpy` D2D each grad into the flat buffer at its offset.
4. Single `all_reduce_async` + `wait` on the entire buffer.
5. `cudaMemcpy` D2D back to each grad's original location.
6. `cudaFree` the flat buffer.
7. Scale all replicated grads by `1/world_size`.

---

## 5. Fused cross-entropy AllReduces with pre-allocated buffer

**File:** `CustomDNN/CustomDNN.cpp` (line ~678)

**What:** Fused two independent AllReduce calls (`local_sum_exp` and `local_target_logit`, both `[BT]` tensors) into a single AllReduce on a `[2*BT]` buffer. The buffer is pre-allocated as a `static` variable, reused across all micro-steps.

**Why:** Two separate AllReduces double the NCCL launch overhead. Since both are SUM reductions on same-sized `[BT]` tensors, they can be concatenated and reduced in one call. Additionally, the buffer is `static` with lazy resize — allocated once on first call and reused, avoiding `cudaMalloc/cudaFree` overhead on every micro-step (which runs `grad_accum_steps` times per optimizer step).

**How:**
1. Static `float* fused_buf` and `int64_t fused_buf_size` persist across calls.
2. On first call (or if `BT` grows), allocate/reallocate the buffer.
3. Pack `local_sum_exp` at offset 0 and `local_target_logit` at offset `BT`.
4. Single `all_reduce_async` + `wait` on `2*BT` elements.
5. Unpack results back to the original tensors.

---

# Phase 3: Memory Allocations & Hot Path Safety

## 6. Reuse DLinear output buffer

**Files:**
- `CustomDNN/CustomDNN.h` (line ~264) — added `cached_output_` member
- `CustomDNN/CustomDNN.cpp` (line ~165) — lazy init and reuse

**What:** Added a `std::unique_ptr<DTensor> cached_output_` member to `DLinear`. On the first `forward()` call, the output DTensor is constructed and stored. On subsequent calls, the same DTensor is reused.

**Why:** The DTensor constructor is expensive — it calls `MPI_Comm_rank`, `cudaGetDeviceCount`, `cudaSetDevice`, and allocates a full GPU tensor via `Tensor::randn` every time. Since every model has multiple DLinear layers, each called every micro-step, this overhead compounds significantly. The output shape never changes, so reusing the same DTensor avoids all that repeated setup.

**How:** Lazy initialization with `if (!cached_output_)` on first call, then `DTensor& output = *cached_output_` on every call.

---

## 7. Removed `static` from `grad_scale`

**File:** `CustomDNN/gpt2_tests/gpt2_tp_test.cpp` (line ~601)

**What:** Changed `grad_scale` from `static Tensor` to a local `Tensor`, allocated once per optimizer step.

**Why:** A `static` tensor persists for the process lifetime and is shared across all calls. This is unsafe if `grad_accum_steps` or the device ever changes, and could cause subtle bugs in multi-model setups. The cost of allocating a 1-element tensor once per step is negligible.

**How:** Removed the `static` keyword.

---

## 8. Documented `d_layer_norms` static assumption

**File:** `CustomDNN/CustomDNN.cpp` (line ~34)

**What:** Added a comment documenting that the static GPU buffers in `clip_grad_norm_dtensor_nccl` assume single-model usage.

**Why:** The static buffers are a valid optimization (avoids cudaMalloc/cudaFree per step, grows lazily). However, they would cause data races or incorrect results if two model instances called this function concurrently. The comment makes this assumption explicit for future maintainers.

---

# Phase 4: Kernels and Advanced Optimizations

## 9. `.contiguous()` calls — KEPT (not removable)

**File:** `CustomDNN/gpt2_tests/gpt2_tp_test.cpp` (lines 122-124)

**Finding:** Investigation confirmed that while the matmul kernel fully supports strided tensors (uses stride-aware cuBLAS dispatch), the **softmax kernel assumes contiguous memory** — it indexes as `row * cols + col`. Removing `.contiguous()` after transpose would produce incorrect softmax results.

**Action:** No change. Removing `.contiguous()` requires rewriting the softmax (and tril) kernels to be stride-aware. Deferred to a future PR.

---

## 10. Documented broken DBlock::forward

**File:** `CustomDNN/gpt2_tests/gpt2_tp_test.cpp` (line ~185)

**What:** Cleaned up the `DBlock::forward` implementation — removed the dead `ln_1_->forward()` call (whose result was unused) and added a clear WARNING comment explaining the limitation.

**Why:** The function only runs the MLP branch, completely skipping attention. The GPT training loop drives attention and MLP separately (for compute/communication overlap), so this function is never called in practice. However, `test.cpp` instantiates `DBlock`, so the override must exist to satisfy the virtual interface.

**How:** Removed the unused `ln_1_->forward()` call and added a comment documenting that this is an MLP-only placeholder.

---

## 11. Fused dual LayerNorm — DEFERRED

**Finding:** The two `ln1`/`ln2` forward calls on the same input `x` each read the full tensor, compute mean/variance, normalize, and write output independently. A fused kernel would halve memory bandwidth by reading `x` once.

**Action:** Deferred. This requires a custom CUDA kernel in the Tensor-Implementations layer that computes two independent LayerNorm outputs in a single pass. The existing LayerNorm forward delegates to a generic `nn::LayerNorm` class — fusing would need a new kernel alongside the existing one. Recommended as a standalone PR.
