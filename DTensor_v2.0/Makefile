CXX = /home/blu-bridge25/.local/bin/mpic++

NVCC_LINKER = /usr/local/cuda/bin/nvcc

CXXFLAGS += -fPIC -std=c++17 -g

NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -g -ccbin=/home/blu-bridge25/.local/bin/mpic++

TARGET = main

CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := /home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor_v2.0/Tensor-Implementations
TENSOR_LIB_A   := /home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor_v2.0/Tensor-Implementations/lib/libtensor.a

INCLUDES = \
    -I. \
    -I./tensor \
    -I./process_group \
    -I./memory \
    -I./bridge \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include

LIB_PATHS = \
    -L/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor_v2.0/Tensor-Implementations/lib \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib

LIBS = -lmpi -lnccl -lcudart -lcublas -lcurand -lgomp -lstdc++

LDFLAGS = -L/home/blu-bridge25/.local/lib -Xlinker -rpath -Xlinker /home/blu-bridge25/.local/lib

SRCS = main.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNCCL_new.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp

OBJS = $(SRCS:.cpp=.o)

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
	test_mlp_forward.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNCCL_new.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \

TEST_MLP_OBJS = $(TEST_MLP_SRCS:.cpp=.o)

# test_mlp_forward: $(TEST_MLP_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating test executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(TEST_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Test build complete."


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

CUDA_SRCS = process_group/reverse.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

TP_MLP_SRCS = \
    examples/tensor_parallel_mlp.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl_new.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp 


TP_MLP_OBJS = $(TP_MLP_SRCS:.cpp=.o) $(CUDA_OBJS)

CUDA_LDFLAGS = -L/usr/local/cuda-12.6/include

%.o: %.cu
	nvcc -c $< -o $@

# tensor_parallel_mlp: $(TP_MLP_OBJS) $(TENSOR_LIB_A)
# 	@echo "\n[LINKING] Creating example executable: $@"
# 	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LDFLAGS) -o examples/$@ $(TP_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
# 	@echo "[SUCCESS] Tensor Parallel MLP Example build complete."


tensor_parallel_mlp: $(TP_MLP_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(TP_MLP_OBJS) $(TENSOR_LIB_A) $(LIBS) $(LDFLAGS)



# Use a more explicit pattern that handles subdirectories better

$(TP_MLP_SRCS:.cpp=.o): %.o: %.cpp
	@echo "[COMPILE C++] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# NEW CUDA rule
$(CUDA_OBJS): %.o: %.cu
	@echo "[COMPILE CUDA] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Shard Benchmarking
SHARD_BENCH_SRCS = \
    benchmarks/shard_benchmarking.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl_new.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
    tensor/placement.cpp

SHARD_BENCH_OBJS = $(SHARD_BENCH_SRCS:.cpp=.o) $(CUDA_OBJS)

shard_benchmarking: $(SHARD_BENCH_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o benchmarks/$@ $(SHARD_BENCH_OBJS) $(TENSOR_LIB_A) $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Shard Benchmarking build complete."
	@echo "\nRun with: mpirun -np 2 ./benchmarks/shard_benchmarking\n"

# # Striped Attention Example
# STRIPED_ATTN_SRCS = \
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