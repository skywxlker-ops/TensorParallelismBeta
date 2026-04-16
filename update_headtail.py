with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/headtail_kernel.cuh', 'r') as f:
    cuh = f.read()

cuh += "\n"
cuh += "void launch_headtail_partial_update(\n"
cuh += "    const void* src,\n"
cuh += "    void* dst,\n"
cuh += "    int64_t outer_size,\n"
cuh += "    int64_t seq_len_full,\n"
cuh += "    int64_t inner_size,\n"
cuh += "    int64_t rank,\n"
cuh += "    int64_t world_size,\n"
cuh += "    int64_t elem_bytes,\n"
cuh += "    bool add,\n"
cuh += "    cudaStream_t stream);\n"

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/headtail_kernel.cuh', 'w') as f:
    f.write(cuh)

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/headtail_kernel.cu', 'r') as f:
    cu = f.read()

new_kernel = """
// ---------------------------------------------------------------------------
// HeadTail partial update kernel (dtype-agnostic but supports float32 addition)
// ---------------------------------------------------------------------------
__global__ void headtail_partial_update_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int64_t outer_size,
    int64_t seq_len_full,
    int64_t inner_size,
    int64_t rank,
    int64_t world_size,
    int64_t elem_bytes,
    bool add,
    int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t S = seq_len_full / world_size;
    int64_t inner_idx = idx % inner_size;
    int64_t temp = idx / inner_size;
    int64_t k = temp % S;
    int64_t outer_idx = temp / S;

    int64_t half = S / 2;
    int64_t d;
    if (k < half) {
        d = rank * half + k;
    } else {
        d = seq_len_full - (rank + 1) * half + (k - half);
    }

    int64_t dst_off = (outer_idx * (seq_len_full * inner_size) + d * inner_size + inner_idx) * elem_bytes;
    int64_t src_off = idx * elem_bytes;

    if (add && elem_bytes == sizeof(float)) {
        float* dst_f = (float*)(dst + dst_off);
        const float* src_f = (const float*)(src + src_off);
        *dst_f += *src_f;
    } else {
        for (int64_t b = 0; b < elem_bytes; ++b) {
            dst[dst_off + b] = src[src_off + b];
        }
    }
}

void launch_headtail_partial_update(
    const void* src,
    void* dst,
    int64_t outer_size,
    int64_t seq_len_full,
    int64_t inner_size,
    int64_t rank,
    int64_t world_size,
    int64_t elem_bytes,
    bool add,
    cudaStream_t stream)
{
    int64_t S = seq_len_full / world_size;
    int64_t total = outer_size * S * inner_size;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    headtail_partial_update_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const char*>(src),
        static_cast<char*>(dst),
        outer_size, seq_len_full, inner_size, rank, world_size, elem_bytes, add, total);
}
"""

cu += new_kernel

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/headtail_kernel.cu', 'w') as f:
    f.write(cu)
