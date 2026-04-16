with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/dtensor.h', 'r') as f:
    h = f.read()

# Add partial_update to HeadTail class
if "void partial_update(" not in h:
    h = h.replace("void loadbalance(Tensor& tensor);", "void loadbalance(Tensor& tensor);\n    void partial_update(const Tensor& src, Tensor& dst, int rank, bool add = false);")
    with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/dtensor.h', 'w') as f:
        f.write(h)

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/dtensor.cpp', 'r') as f:
    cpp = f.read()

impl = """
void HeadTail::partial_update(const Tensor& src, Tensor& dst, int rank, bool add) {
    int64_t seq_len_full = dst.shape().dims[chunk_dim_];
    int64_t outer_size = 1;
    for (int i = 0; i < chunk_dim_; ++i) {
        outer_size *= src.shape().dims[i];
    }
    int64_t inner_size = 1;
    for (size_t i = chunk_dim_ + 1; i < src.shape().dims.size(); ++i) {
        inner_size *= src.shape().dims[i];
    }
    int64_t elem_bytes = src.dtype() == Dtype::Float32 ? 4 : 8; // Adjust based on your types
    if (src.dtype() == Dtype::Float16 || src.dtype() == Dtype::BFloat16) elem_bytes = 2;

    launch_headtail_partial_update(
        src.data_ptr(), dst.data_ptr(),
        outer_size, seq_len_full, inner_size,
        rank, world_size_, elem_bytes, add, stream_
    );
}
"""

if "void HeadTail::partial_update" not in cpp:
    cpp += impl
    with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/tensor/dtensor.cpp', 'w') as f:
        f.write(cpp)
