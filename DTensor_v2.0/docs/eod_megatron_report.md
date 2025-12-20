# EOD Report: MegatronLM Tensor Parallelism Features

**Date:** December 20, 2024

---

## Summary

Today I analyzed Megatron-Core's tensor parallelism implementation to find TP-specific features we're missing in DTensor.

---

## Gap Analysis (TP Features Only)

| Feature | Us | Megatron | Priority |
|---------|:--:|:--------:|:--------:|
| Column-Parallel Linear | ✅ | ✅ | Done |
| Row-Parallel Linear | ✅ | ✅ | Done |
| Fused matmul+activation | ✅ | ✅ | Done |
| **Sequence Parallelism** | ❌ | ✅ | High |
| **VocabParallelEmbedding** | ❌ | ✅ | High |
| **Async Comm Overlap** | ⚠️ | ✅ | Medium |

---

## 1. Sequence Parallelism (Priority: HIGH)

**The Problem:** Between parallel layers, activations are `[S, B, H]` on EVERY GPU - redundant memory.

**Megatron's Solution:** Split sequence dimension, use AllGather before matmul, ReduceScatter after:

```python
# megatron/core/tensor_parallel/layers.py (line 998-1003)
if sequence_parallel:
    dim_size[0] = dim_size[0] * tp_group.size()
    all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
    dist_all_gather_func(all_gather_buffer, input, group=tp_group)
    total_input = all_gather_buffer
```

And after the column-parallel matmul, reduce-scatter back:

```python
# megatron/core/tensor_parallel/layers.py (line 1056-1064)
if ctx.sequence_parallel:
    sub_grad_input = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
    handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)
```

**What we'd add:**

```cpp
DTensor DTensor::_column_parallel_matmul_sp(const DTensor& other) const {
    // AllGather input: [S/P, B, H] -> [S, B, H]
    DTensor input_full = *this;
    input_full.allGather();
    
    // Local matmul
    OwnTensor::Tensor Y = TensorOpsBridge::matmul(input_full.tensor_, other.tensor_);
    
    // ReduceScatter output: [S, B, H/P] -> [S/P, B, H/P]
    DTensor result(device_mesh_, pg_, Y, output_layout);
    result.reduceScatter();
    return result;
}
```

**Memory savings:** Activation memory reduced by `world_size` factor.

---

## 2. VocabParallelEmbedding (Priority: HIGH)

**The Problem:** Embedding table is huge (50k vocab × 4096 hidden = 800MB per GPU).

**Megatron's Solution:** Each GPU holds `vocab_size/P` rows, masks out-of-range tokens:

```python
# megatron/core/tensor_parallel/layers.py (line 826-856)
def forward(self, input_):
    # Each GPU only handles tokens in its vocab range
    input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
    masked_input = input_.clone() - self.vocab_start_index
    masked_input[input_mask] = 0
    
    # Lookup in local embedding shard
    output_parallel = F.embedding(masked_input, self.weight)
    output_parallel[input_mask, :] = 0.0  # Zero invalid positions
    
    # AllReduce combines results (only one GPU has valid lookup per token)
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output
```

**What we'd add:**

```cpp
class VocabParallelEmbedding {
    DTensor weight_;  // [vocab_size/P, hidden]
    int vocab_start_, vocab_end_;
    
    DTensor forward(const std::vector<int>& token_ids) {
        // 1. Mask tokens outside [vocab_start_, vocab_end_)
        // 2. Lookup local embeddings (offset by vocab_start_)
        // 3. Zero out masked positions  
        // 4. AllReduce to combine
        result.allReduce();
        return result;
    }
};
```

---

## 3. Communication-Computation Overlap (Priority: MEDIUM)

**The Problem:** Sequential comm blocks compute.

**Megatron uses CUDA_DEVICE_MAX_CONNECTIONS=1:** This forces kernel launch order, enabling predictable overlap:

```python
# megatron/core/tensor_parallel/layers.py (line 1035-1040)
handle = dist_all_gather_func(all_gather_buffer, input, group=tp_group, async_op=True)
# Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
# gather is scheduled before the input gradient computation
```

**We already have this infra:**

```cpp
// Our multi-stream setup enables this:
cudaEventRecord(compute_event_, compute_stream_);
cudaStreamWaitEvent(comm_stream_, compute_event_, 0);
pg_->all_reduce_async(...);  // Runs on comm_stream_
// compute_stream_ can continue other work!
```

**Gap:** We have the primitives, need to wire them into matmul paths.

---

## 4. Global Memory Buffer (Optimization)

**Megatron's trick:** Single shared buffer for all AllGather/ReduceScatter ops:

```python
# megatron/core/tensor_parallel/layers.py (line 1001)
all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
```

**Benefit:** No per-op allocation. We have `CachingAllocator` but could add a dedicated TP buffer.

---

## Key Insights

1. **Sequence parallelism** is the biggest memory win - we have AllGather/ReduceScatter, just need to wire them
2. **VocabParallelEmbedding** is critical for LLM work - straightforward with our existing primitives
3. **We have multi-stream** - need to integrate async ops into parallel matmul paths
4. **Global buffer** - minor optimization, our caching allocator mostly covers this

---

## Next Steps

- [ ] Add `sequence_parallel` flag to column/row parallel matmul
- [ ] Implement `VocabParallelEmbedding` class
- [ ] Wire async collectives into matmul paths for overlap

---
*~2 hours*
