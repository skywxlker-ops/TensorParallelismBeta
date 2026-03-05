# Checkpointing Test Suite Documentation

This document summarizes the comprehensive test suite for gradient checkpointing in `agtensor`. The tests ensure numerical correctness, memory efficiency, RNG reproducibility, and robustness across various graph topologies on both CPU and GPU.

## General Correctness & Topology Tests
Files: [edge_cases.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/edge_cases.cpp), [edge_cases_gpu.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/edge_cases_gpu.cpp), [rigorous.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/rigorous.cpp), [rigorous_v2.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/rigorous_v2.cpp)

- **Unused Inputs**: Verifies that tensors requiring gradients but not used in the checkpointed block do not cause issues or incorrect gradient accumulation.
- **Multiple Outputs**: Ensures that blocks returning multiple tensors correctly handle gradients from all output paths during recomputation.
- **Divergent Paths**: Test cases where an input is used both inside and outside a checkpointed block.
- **Chained Checkpoints**: Multiple sequential checkpointed blocks to ensure gradients flow correctly through successive recomputation stages.
- **Shared Inputs**: Handles cases where the same tensor is passed multiple times as input to a checkpointed block.
- **Complex Graphs (Diamond Pattern)**: Blocks with internal branching and merging (e.g., `x^2 + 2x`) to verify internal recomputation correctness.
- **Mixed Grad Requirements**: Blocks with a mix of inputs that do and do not require gradients.
- **No-Grad Inputs**: Verifies that blocks with no inputs requiring gradients skip unnecessary recomputation logic.

## RNG & Stochasticity
Files: [rng_complex.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/rng_complex.cpp), [rng_phase.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/rng_phase.cpp), [rng_complex.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/rng_complex.cpp)

- **State Capture/Restore**: Basic and multi-snapshot RNG state management for CPU and GPU.
- **Stochastic Consistency**: Ensures that dropout-like operations or noise addition produce bit-identical results during the forward pass and the recomputed backward pass.
- **Distribution Parity**: Verifies that `rand`, `randn`, and uniform distributions maintain consistency across recomputations.
- **RAII Guards**: Tests for `RNGStateGuard` to ensure state is correctly restored even if recomputation is interrupted.

## Memory Optimization & Profiling
Files: [with.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/with.cpp), [without.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/without.cpp), [gpu_with.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/gpu_with.cpp), [gpu_without.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/gpu_without.cpp), [manual.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/manual.cpp)

- **Baseline Comparison**: Direct memory usage comparison (using Valgrind Massif or GPU memory info) between standard forward passes (storing all activations) and checkpointed passes.
- **Activation Freeing**: Verification that intermediate activations are indeed released from memory immediately after the forward block finishes.
- **Manual Memory Accounting**: Tracking tensor `nbytes()` and total active tensor counts to quantify savings.

## Sequential Checkpointing
Files: [uniform.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/uniform.cpp), [uniform_edge_cases.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/uniform_edge_cases.cpp)

- **Automatic Segmenting**: Tests for `checkpoint_sequential` which automatically divides a `nn::Sequential` model into segments.
- **Segment Granularity**: Verifies behavior with 1 segment (full model), max segments (one per module), and edge cases like zero or over-sized segment counts.

## Robustness & Interaction
Files: [test_checkpoint_exception.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/test_checkpoint_exception.cpp), [grad_mode.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/grad_mode.cpp), [training_loop.cpp](file:///home/blubridge-029/agtensor/tensor/Tests/Checkpointing/training_loop.cpp)

- **Exception Safety**: Ensures that if recomputation fails (e.g., out of memory or logic error), the `CheckpointNode` still correctly releases any partially allocated saved variables using RAII.
- **Grad Mode Interaction**: Verifies that `NoGradGuard` works correctly with checkpointing and that recomputation happens with gradients enabled even if the outer scope is `no_grad`.
- **Training Loops**: 10-layer MLP training stability over multiple epochs with parameter updates.

## Suggested Production-Ready Tests
To ensure the checkpointing system is fully robust for a production environment and can withstand rigorous code review, the following additional tests are recommended:

### 1. In-place Operation Safeguards
- **Problem**: If a checkpointed block modifies its inputs or any saved tensors in-place, the forward recomputation will use corrupted data, leading to silent numerical errors.
- **Test**: Create a block that performs an in-place operation (e.g., `x.add_(1)`) on an input. Verify that the system either throws an error (preferred) or correctly handles the versioning of the tensor to ensure recomputation uses the original state.

### 2. Mixed Precision & Scaled Gradients
- **Problem**: Many production models use `float16` or `bfloat16`. Checkpointing must ensure and verify that recomputation doesn't introduce precision loss beyond the expected limits of these types.
- **Test**: Run parity tests with `float16` and ensure that the gradient scaling (if used) is applied correctly during the recomputed backward pass.

### 3. Distributed & Multi-GPU Execution
- **Problem**: In distributed settings (DDP/HPU), gradient synchronization happens at specific points. Checkpointing changes the timing of when gradients are available.
- **Test**: Verify that checkpointing works in a multi-device setup and that gradients are synchronized correctly for all parameters, including those inside checkpointed segments.

### 4. Dynamic Shapes & Control Flow
- **Problem**: Models with dynamic input shapes (e.g., variable sequence lengths) need to ensure that the recomputation block correctly handles the shape it was originally called with.
- **Test**: Run a checkpointed block with inputs of varying shapes in the same training loop and verify recomputation consistency for each shape.

### 5. Memory Fragmentation & Long-term Stability
- **Problem**: Constant allocation and deallocation of intermediate activations can lead to memory fragmentation.
- **Test**: Run a very long training loop (1000+ iterations) with large tensors and monitor if the "Free Memory" on GPU/CPU stays stable or if it gradually decreases due to fragmentation.

### 6. Autograd Hook Interactions
- **Problem**: Users often register `backward_hooks` or `post_accumulate_grad_hooks`.
- **Test**: Register hooks on tensors used within a checkpointed block and verify they are called exactly once (or as expected) and with correct values during the recomputed backward pass.

### 7. Re-entrancy Stress Test
- **Problem**: Our `Engine` was recently made re-entrant for sequential backward.
- **Test**: Deeply nested checkpoints (e.g., Level 5 nesting) with complex branching to ensure the re-entrancy logic handles high stack depths and complex graph dependencies without deadlocks or state corruption.
