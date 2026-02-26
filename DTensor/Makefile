CXX = /home/blu-bridge25/.local/bin/mpic++

NVCC_LINKER = /usr/local/cuda/bin/nvcc
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -G -arch=sm_86 -ccbin=$(CXX)

CXXFLAGS += -fPIC -std=c++17 -O3 -g -DWITH_CUDA

NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -G -arch=sm_86 -ccbin=/home/blu-bridge25/.local/bin/mpic++

TARGET = main

CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := /home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Tensor-Implementations
TENSOR_LIB_A   := /home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Tensor-Implementations/lib/libtensor.a

INCLUDES = \
    -I. \
    -I./tensor \
    -I./process_group \
    -I./dnn \
    -I./memory \
    -I./bridge \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include \
#     -I/home/blu-bridge25/Study/Code/TensorParallelismBeta/autograd/cgadimpl/include \
    $(AUTOGRAD_INCLUDES)

LIB_PATHS = \
    -L/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Tensor-Implementations/lib \
#     -L/home/blu-bridge25/Study/Code/TensorParallelismBeta/autograd/cgadimpl/build \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
    $(AUTOGRAD_LIB_PATHS)

LIBS = -lmpi -lnccl -lcudart -lcublas -lcurand -lgomp -lstdc++ $(AUTOGRAD_LIBS) -lz -lpthread -ldl

LDFLAGS = -L/home/blu-bridge25/.local/lib -Xlinker -rpath -Xlinker /home/blu-bridge25/.local/lib

SRCS = main.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp

OBJS = $(SRCS:.cpp=.o) $(CUDA_OBJS) 

.PHONY: all clean test test_mlp_forward

all: $(TARGET)

# test: test_mlp_forward
# 	@echo "\n[TEST BUILD COMPLETE]"
# 	@echo "Run the test with: mpirun -np 2 ./test_mlp_forward"
# 	@echo ""

$(TARGET): $(OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -L/home/blu-bridge25/.local/lib -Xlinker -rpath -Xlinker /home/blu-bridge25/.local/lib -o $(TARGET) $(OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS) -Xlinker --no-keep-memory -Xlinker --reduce-memory-overheads
	@echo "[SUCCESS] Build complete."


TEST_MLP_SRCS = \
	tests/kiruthik_mlp_test.cpp \
# 	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \

TEST_MLP_OBJS = $(TEST_MLP_SRCS:.cpp=.o)

kkk: $(TEST_MLP_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(TEST_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."


%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# benchmarks/%.o: benchmarks/%.cpp
# 	@echo "[COMPILE] $<"
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# tests/%.o: tests/%.cpp
# 	@echo "[COMPILE] $<"
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the '/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor_v2.0/Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi


clean:
	@echo "[CLEAN] Removing object files and executables..."
	rm -f $(OBJS) $(TARGET) $(TEST_MLP_OBJS) test_mlp_forward
	rm -f tests/*.o tests/test_device_mesh tests/test_matmul
	rm -f TrainingScripts/*.o
	rm -f benchmarks/*.o benchmarks/nccl_benchmark benchmarks/matmul_benchmark benchmarks/row_parallel_breakdown benchmarks/row_parallel_breakdown_custom benchmarks/shard_benchmarking
	rm -f tensor/dtensor_custom.o bridge/tensor_ops_bridge_custom.o

.PHONY: test_device_mesh nccl_benchmark test_matmul matmul_benchmark row_parallel_breakdown row_parallel_breakdown_custom gradient_sync_example shard_benchmarking

# TEST_DEVICE_MESH_SRCS = \
# 	tests/test_device_mesh.cpp \
# 	tensor/dtensor.cpp \
# 	tensor/device_mesh.cpp \
# 	process_group/process_group_nccl.cpp \
# 	memory/cachingAllocator.cpp \
# 	bridge/tensor_ops_bridge.cpp 

# TEST_DEVICE_MESH_OBJS = $(TEST_DEVICE_MESH_SRCS:.cpp=.o)

# test_device_mesh: $(TEST_DEVICE_MESH_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating test executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_DEVICE_MESH_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Test build complete."
# 	@echo "\nRun : mpirun -np 2 ./tests/test_device_mesh\n"

# NCCL_BENCHMARK_SRCS = benchmarks/nccl_benchmark_1.cpp
# NCCL_BENCHMARK_OBJS = $(NCCL_BENCHMARK_SRCS:.cpp=.o)

# nccl_benchmark: $(NCCL_BENCHMARK_OBJS)
# 	@echo "\n[LINKING] Creating benchmark executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(NCCL_BENCHMARK_OBJS) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Benchmark build complete."
# 	@echo "\nRun : mpirun -np 2 ./benchmarks/nccl_benchmark\n"

# MATMUL_BENCHMARK_SRCS = \
# 	benchmarks/matmul_benchmark.cpp \
# 	tensor/dtensor.cpp \
# 	tensor/device_mesh.cpp \
# 	process_group/process_group_nccl.cpp \
# 	memory/cachingAllocator.cpp \
# 	bridge/tensor_ops_bridge.cpp 

# MATMUL_BENCHMARK_OBJS = $(MATMUL_BENCHMARK_SRCS:.cpp=.o)

# matmul_benchmark: $(MATMUL_BENCHMARK_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating benchmark executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(MATMUL_BENCHMARK_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] MatMul Benchmark build complete."
# 	@echo "\nRun : mpirun -np 2 ./benchmarks/matmul_benchmark\n"


# ROW_BREAKDOWN_SRCS = \
# 	benchmarks/row_parallel_breakdown.cpp \
# 	tensor/dtensor.cpp \
# 	tensor/device_mesh.cpp \
# 	process_group/process_group_nccl.cpp \
# 	memory/cachingAllocator.cpp \
# 	bridge/tensor_ops_bridge.cpp 

# ROW_BREAKDOWN_OBJS = $(ROW_BREAKDOWN_SRCS:.cpp=.o)

# row_parallel_breakdown: $(ROW_BREAKDOWN_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating benchmark executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(ROW_BREAKDOWN_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Row-Parallel Breakdown build complete."
# 	@echo "\nRun : mpirun -np 2 ./benchmarks/row_parallel_breakdown\n"


# TEST_MATMUL_SRCS = \
# 	tests/test_matmul.cpp \
# 	tensor/dtensor.cpp \
# 	tensor/device_mesh.cpp \
# 	process_group/process_group_nccl.cpp \
# 	memory/cachingAllocator.cpp \
# 	bridge/tensor_ops_bridge.cpp 

# TEST_MATMUL_OBJS = $(TEST_MATMUL_SRCS:.cpp=.o)

# test_matmul: $(TEST_MATMUL_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating test executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_MATMUL_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] MatMul test build complete."
# 	@echo "\nRun : mpirun -np 2 ./tests/test_matmul\n"

# ROW_BREAKDOWN_CUSTOM_SRCS = \
# 	benchmarks/row_parallel_breakdown.cpp \
# 	tensor/dtensor_own.cpp \
# 	tensor/device_mesh.cpp \
# 	process_group/process_group_nccl.cpp \
# 	memory/cachingAllocator.cpp \
# 	bridge/tensor_ops_bridge_own.cpp 

# ROW_BREAKDOWN_CUSTOM_OBJS = $(ROW_BREAKDOWN_CUSTOM_SRCS:.cpp=.o)

# row_parallel_breakdown_custom: $(ROW_BREAKDOWN_CUSTOM_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating custom benchmark executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(ROW_BREAKDOWN_CUSTOM_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Custom Row-Parallel Breakdown build complete."
# 	@echo "\nRun : mpirun -np 2 ./benchmarks/row_parallel_breakdown_custom\n"

# GRADIENT_SYNC_SRCS = \
# 	examples/gradient_sync_example.cpp \
# 	tensor/dtensor.cpp \
# 	tensor/device_mesh.cpp \
# 	process_group/process_group_nccl.cpp \
# 	memory/cachingAllocator.cpp \
# 	bridge/tensor_ops_bridge.cpp 

# GRADIENT_SYNC_OBJS = $(GRADIENT_SYNC_SRCS:.cpp=.o)

# examples/%.o: examples/%.cpp
# 	@echo "[COMPILE] $<"
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# gradient_sync_example: $(GRADIENT_SYNC_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating example executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o examples/$@ $(GRADIENT_SYNC_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Gradient Sync Example build complete."
# 	@echo "\nRun with 4 GPUs: mpirun -np 4 ./examples/gradient_sync_example\n"

# Tensor Parallel MLP Example

CUDA_SRCS =  process_group/fused_transpose_kernel.cu process_group/shard_fused_transpose_kernel.cu dnn/dist_grad_norm_kernels.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

TP_MLP_SRCS = \
    examples/tensor_parallel_mlp_copy.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp 


TP_MLP_OBJS = $(TP_MLP_SRCS:.cpp=.o) $(CUDA_OBJS)

CUDA_LDFLAGS = -L/usr/local/cuda-12.6/include

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

TrainingScripts/EntropyKernels.o: TrainingScripts/EntropyKernels.cu
	@echo "[COMPILE CUDA] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# tensor_parallel_mlp: $(TP_MLP_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating example executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LDFLAGS) -o examples/$@ $(TP_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Tensor Parallel MLP Example build complete."

# Tensor Parallel MLP with Autograd support (uses sharding and sync)
TP_MLP_AUTOGRAD_SRCS = \
    examples/tensor_parallel_mlp.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp

TP_MLP_AUTOGRAD_OBJS = $(TP_MLP_AUTOGRAD_SRCS:.cpp=.o) $(CUDA_OBJS)

tensor_parallel_mlp: $(TP_MLP_AUTOGRAD_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(TP_MLP_AUTOGRAD_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Tensor Parallel MLP Example with Autograd build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/tensor_parallel_mlp\n"

# Normal MLP with fixed seed data (no sharding/sync) for comparison
MLP_SEED_SRCS = \
    examples/mlp_seed.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp

MLP_SEED_OBJS = $(MLP_SEED_SRCS:.cpp=.o) $(CUDA_OBJS)

mlp_seed: $(MLP_SEED_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(MLP_SEED_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Normal MLP (Fixed Seed) build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/mlp_seed\n"

# Tensor Parallel MLP with fixed seed data (sharding/sync) for comparison
TP_MLP_SEED_SRCS = \
    examples/tensor_parallel_mlp_seed.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp

TP_MLP_SEED_OBJS = $(TP_MLP_SEED_SRCS:.cpp=.o) $(CUDA_OBJS)

tensor_parallel_mlp_seed: $(TP_MLP_SEED_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(TP_MLP_SEED_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Tensor Parallel MLP (Fixed Seed) build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/tensor_parallel_mlp_seed\n"

# DModule MLP Example (uses DistributedNN.h)
DMLP_EXAMPLE_SRCS = \
    examples/dmlp_example.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp

DMLP_EXAMPLE_OBJS = $(DMLP_EXAMPLE_SRCS:.cpp=.o) $(CUDA_OBJS)

dmlp_example: $(DMLP_EXAMPLE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(DMLP_EXAMPLE_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DModule MLP Example build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/dmlp_example\n"

# tensor_parallel_mlp: $(TP_MLP_OBJS) $(TENSOR_LIB_A) $(CGADIMPL_LIB)
# 	@echo "\n[LINKING] $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(TP_MLP_OBJS) -Xlinker --start-group $(CGADIMPL_LIB) $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)




# Use a more explicit pattern that handles subdirectories better

# $(TP_MLP_SRCS:.cpp=.o): %.o: %.cpp
# 	@echo "[COMPILE C++] $<"
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# # NEW CUDA rule  
$(CUDA_OBJS): %.o: %.cu
	@echo "[COMPILE CUDA] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Shard Benchmarking
SHARD_BENCH_SRCS = \
    benchmarks/shard_benchmarking.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
#     memory/cachingAllocator.cpp \
#     bridge/tensor_ops_bridge.cpp \
    tensor/placement.cpp

SHARD_BENCH_OBJS = $(SHARD_BENCH_SRCS:.cpp=.o) $(CUDA_OBJS)

shard_benchmarking: $(SHARD_BENCH_OBJS) $(TENSOR_LIB_A) 
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o benchmarks/$@ $(SHARD_BENCH_OBJS) -Xlinker --start-group  $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Shard Benchmarking build complete."
	@echo "\nRun with: mpirun -np 2 ./benchmarks/shard_benchmarking\n"

# Interconnect Benchmarks
INTERCONNECT_BENCH_SRCS = \
    benchmarks/interconnect_benchmarks.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
#     memory/cachingAllocator.cpp \
#     bridge/tensor_ops_bridge.cpp \
    tensor/placement.cpp

INTERCONNECT_BENCH_OBJS = $(INTERCONNECT_BENCH_SRCS:.cpp=.o) $(CUDA_OBJS)

interconnect_benchmarks: $(INTERCONNECT_BENCH_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o benchmarks/$@ $(INTERCONNECT_BENCH_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Interconnect Benchmarks build complete."
	@echo "\nRun with: mpirun -np 2 ./benchmarks/interconnect_benchmarks\n"


# # Rotate Benchmarking
# ROTATE_BENCH_SRCS = \
#     benchmarks/Rotate_benchmark.cpp \
#     tensor/dtensor.cpp \
#     tensor/device_mesh.cpp \
#     process_group/processGroupNccl.cpp \
#     memory/cachingAllocator.cpp \
#     bridge/tensor_ops_bridge.cpp \
#     tensor/placement.cpp

# ROTATE_BENCH_OBJS = $(ROTATE_BENCH_SRCS:.cpp=.o) $(CUDA_OBJS)

# rotate_benchmarking: $(ROTATE_BENCH_OBJS) $(TENSOR_LIB_A) $(CGADIMPL_LIB)
# 	@echo "\n[LINKING] Creating benchmark executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o benchmarks/$@ $(ROTATE_BENCH_OBJS) -Xlinker --start-group $(CGADIMPL_LIB) $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
# 	@echo "[SUCCESS] Rotate Benchmarking build complete."
# 	@echo "\nRun with: mpirun -np 2 ./benchmarks/rotate_benchmarking\n"

# # TensorLib Sharding Benchmark
# TENSORLIB_BENCH_SRCS = benchmarks/tensorlib_sharding_benchmark.cpp
# TENSORLIB_BENCH_OBJS = $(TENSORLIB_BENCH_SRCS:.cpp=.o)

# benchmarks/tensorlib_sharding_benchmark.o: benchmarks/tensorlib_sharding_benchmark.cpp
# 	@echo "[COMPILE] $<"
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# tensorlib_sharding_benchmark: benchmarks/tensorlib_sharding_benchmark.o $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating benchmark executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o benchmarks/$@ benchmarks/tensorlib_sharding_benchmark.o $(TENSOR_LIB_A) $(LIBS) $(LDFLAGS)
# 	@echo "[SUCCESS] TensorLib Sharding Benchmark build complete."
# 	@echo "\nRun with: mpirun -np 2 ./benchmarks/tensorlib_sharding_benchmark\n"

# # LibTorch Sharding Benchmark (requires LibTorch/PyTorch installation)
# TORCH_DIR = /usr/local/lib/python3.10/dist-packages/torch
# TORCH_INCLUDES = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
# TORCH_LIBS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,-rpath,$(TORCH_DIR)/lib

# benchmarks/libtorch_sharding_benchmark.o: benchmarks/libtorch_sharding_benchmark.cpp
# 	@echo "[COMPILE] $<"
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TORCH_INCLUDES) -c $< -o $@

# libtorch_sharding_benchmark: benchmarks/libtorch_sharding_benchmark.o
# 	@echo "\n[LINKING] Creating benchmark executable: $@"
# 	$(CXX) -o benchmarks/$@ benchmarks/libtorch_sharding_benchmark.o $(TORCH_LIBS) -L$(CUDA_HOME)/lib64 -lmpi -lnccl -lcudart $(LDFLAGS)
# 	@echo "[SUCCESS] LibTorch Sharding Benchmark build complete."
# 	@echo "\nRun with: mpirun -np 2 ./benchmarks/libtorch_sharding_benchmark\n"

# # # Striped Attention Example
# # STRIPED_ATTN_SRCS = \
# 	examples/striped_attention_example.cpp \
# 	tensor/dtensor.cpp \
# 	tensor/device_mesh.cpp \
# 	process_group/process_group_nccl.cpp \
# 	memory/cachingAllocator.cpp \
# 	bridge/tensor_ops_bridge.cpp 

# STRIPED_ATTN_OBJS = $(STRIPED_ATTN_SRCS:.cpp=.o)

# striped_attention_example: $(STRIPED_ATTN_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating example executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o examples/$@ $(STRIPED_ATTN_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Striped Attention Example build complete."
# 	@echo "\nRun with 4 GPUs: mpirun -np 4 ./examples/striped_attention_example\n"
# Embedding Example
EMBEDDING_EXAMPLE_SRCS = \
    examples/embedding_example.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

EMBEDDING_EXAMPLE_OBJS = $(EMBEDDING_EXAMPLE_SRCS:.cpp=.o) $(CUDA_OBJS)

embedding_example: $(EMBEDDING_EXAMPLE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(EMBEDDING_EXAMPLE_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Embedding Example build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/embedding_example\n"

# DMLP Embedding Example
DMLP_EMBEDDING_SRCS = \
    examples/dmlp_embedding_example.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

DMLP_EMBEDDING_OBJS = $(DMLP_EMBEDDING_SRCS:.cpp=.o) $(CUDA_OBJS)

dmlp_embedding_example: $(DMLP_EMBEDDING_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(DMLP_EMBEDDING_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DMLP Embedding Example build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/dmlp_embedding_example\n"

# DMLP Full (Parallel Embedding)
DMLP_FULL_SRCS = \
    examples/dmlp_full.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

DMLP_FULL_OBJS = $(DMLP_FULL_SRCS:.cpp=.o) $(CUDA_OBJS)

dmlp_full: $(DMLP_FULL_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(DMLP_FULL_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DMLP Full build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/dmlp_full\n"

# GPT2 Parallel Test
GPT2_PARALLEL_TEST_SRCS = \
    TrainingScripts/gpt2_parallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_PARALLEL_TEST_OBJS = $(GPT2_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS)

gpt2_parallel_test: $(GPT2_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_parallel_test\n"

# GPT2 Entropy Parallel Test
GPT2_ENTROPY_PARALLEL_TEST_SRCS = \
    TrainingScripts/gpt2_entropy_parallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_ENTROPY_PARALLEL_TEST_OBJS = $(GPT2_ENTROPY_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/EntropyKernels.o

gpt2_entropy_parallel_test: $(GPT2_ENTROPY_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_ENTROPY_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Entropy Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_entropy_parallel_test\n"


GPT2_TP_TEST_SRCS = \
    gpt2_tp_test/gpt2_tp_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNCCL.cpp \
    tensor/placement.cpp

GPT2_TP_TEST_OBJS = $(GPT2_TP_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/EntropyKernels.o Tensor-Implementations/src/Kernels/cuda/MultiTensorKernels.o

gpt2_tp_test: $(GPT2_TP_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: gpt2_tp_test_exec"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o gpt2_tp_test_exec $(GPT2_TP_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 TP Test build complete."
	@echo "\nRun with: mpirun -np 2 ./gpt2_tp_test_exec \n"

GPT2_TP_MLP_TEST_SRCS = \
    gpt2_tp_test/gpt2_tp_mlp_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNCCL.cpp \
    tensor/placement.cpp \
    
# GPT2_CUDA_SRCS = dnn/dist_grad_norm_kernels.cu
# CUDA_OBJS = $(GPT2_CUDA_SRCS:.cu = .o)

GPT2_TP_MLP_TEST_OBJS = $(GPT2_TP_MLP_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/EntropyKernels.o  Tensor-Implementations/src/Kernels/cuda/MultiTensorKernels.o


gpt2_tp_mlp_test: $(GPT2_TP_MLP_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: gpt2_tp_mlp_test_exec"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o gpt2_tp_mlp_test_exec $(GPT2_TP_MLP_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 TP MLP Test build complete."
	@echo "\nRun with: mpirun -np 2 ./gpt2_tp_mlp_test_exec \n"

    # Example target for the failing test
# gpt2_tp_mlp_test: gpt2_tp_test/gpt2_tp_mlp_test.o $(CUDA_OBJS) $(OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating example executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o gpt2_tp_mlp_test_exec $(GPT2_TP_MLP_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
#     @echo "[SUCCESS] GPT2 TP MLP Test build complete." 
# 	@echo "\nRun with: mpirun -np 2 ./gpt2_tp_mlp_test_exec \n"

# GPT2_TP_MLP_TEST_SRCS = \
#     gpt2_tp_test/gpt2_tp_mlp_test.cpp \
#     tensor/dtensor.cpp \
#     tensor/device_mesh.cpp \
#     process_group/processGroupNccl.cpp \
#     memory/cachingAllocator.cpp \
#     bridge/tensor_ops_bridge.cpp \
#     tensor/placement.cpp

# GPT2_TP_MLP_TEST_OBJS = $(GPT2_TP_MLP_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/EntropyKernels.o

# gpt2_tp_mlp_test: $(GPT2_TP_MLP_TEST_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating example executable: gpt2_tp_mlp_test_"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o gpt2_tp_mlp_test $(GPT2_TP_MLP_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
# 	@echo "[SUCCESS] GPT2 TP MLP Test build complete."
# 	@echo "\nRun with: mpirun -np 2 ./gpt2_tp_mlp_test \n"

# GPT2 Sync Verification Test
GPT2_SYNC_VERIFY_TEST_SRCS = \
    TrainingScripts/gpt2_sync_verify_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_SYNC_VERIFY_TEST_OBJS = $(GPT2_SYNC_VERIFY_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/MaskOps.o

gpt2_sync_verify_test: $(GPT2_SYNC_VERIFY_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_SYNC_VERIFY_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Sync Verification Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_sync_verify_test\n"

# GPT2 MLP Parallel Test (Non-parallel Embedding/LM Head)
GPT2_MLP_PARALLEL_TEST_SRCS = \
    TrainingScripts/gpt2_mlp_parallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_MLP_PARALLEL_TEST_OBJS = $(GPT2_MLP_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS)

gpt2_mlp_parallel_test: $(GPT2_MLP_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_MLP_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 MLP Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_mlp_parallel_test\n"


# GPT2 Tensor Parallel Test
GPT2_TENSOR_PARALLEL_TEST_SRCS = \
    TrainingScripts/GPT2_TesnorParallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_TENSOR_PARALLEL_TEST_OBJS = $(GPT2_TENSOR_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS)

gpt2_tensor_parallel_test: $(GPT2_TENSOR_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_TENSOR_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Tensor Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_tensor_parallel_test\n"

# DTensor Leak Tests
DTENSOR_LEAK_TESTS_SRCS = \
    tests/dtensor_leak_tests/dtensor_benchmark_main.cpp \
    tests/dtensor_leak_tests/test_dtensor_layers.cpp \
    tests/dtensor_leak_tests/test_dtensor_activations.cpp \
    tests/dtensor_leak_tests/test_dtensor_losses.cpp \
    tests/dtensor_leak_tests/test_gpt2_components.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

DTENSOR_LEAK_TESTS_OBJS = $(DTENSOR_LEAK_TESTS_SRCS:.cpp=.o) $(CUDA_OBJS)

tests/dtensor_leak_tests/%.o: tests/dtensor_leak_tests/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I./tests/dtensor_leak_tests -c $< -o $@

dtensor_leak_tests: $(DTENSOR_LEAK_TESTS_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o tests/dtensor_leak_tests/$@ $(DTENSOR_LEAK_TESTS_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DTensor Leak Tests build complete."
	@echo "\nRun with: mpirun -np 2 ./tests/dtensor_leak_tests/dtensor_leak_tests\n"



# =============================================================================
# Distributed System Rules (Launcher & Sentinel)
# =============================================================================

# Default values if not provided via command line
N ?= 2
PORT ?= 29877

.PHONY: launcher
launcher: $(TENSOR_LIB_A)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify the worker file."; \
		echo "Usage: make launcher FILE=path/to/worker.cpp N=5 PORT=30000"; \
		exit 1; \
	fi
	@echo "--- Step 1: Compiling Worker Snippet: $(FILE) ---"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(INCLUDES) $(FILE) -o worker_bin $(LIB_PATHS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "--- Step 2: Compiling Sentinel Launcher ---"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(INCLUDES) /home/blu-bridge25/TP/TensorParallelismBeta/launcher.cpp -o launcher_bin $(LIB_PATHS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) -lnvidia-ml $(LDFLAGS)
	@echo "--- Step 3: Starting Distributed Cluster (N=$(N), Port=$(PORT)) ---"
	@echo "---------------------------------------------------------"
	./launcher_bin $(N) -- ./worker_bin
	@echo "---------------------------------------------------------"
	@echo "--- Cleaning up binaries ---"
	@rm -f worker_bin launcher_bin