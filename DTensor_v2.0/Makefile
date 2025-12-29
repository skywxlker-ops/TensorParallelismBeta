
CXX = mpic++


NVCC_LINKER = /usr/local/cuda/bin/nvcc

CXXFLAGS = -std=c++17 -O3 -fPIC -g -DWITH_CUDA


NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

TARGET = dtensor_main


CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := ./Tensor-Implementations
TENSOR_LIB_A   := $(TENSOR_LIB_DIR)/lib/libtensor.a


INCLUDES = \
    -I. \
    -I./tensor \
    -I./process_group \
    -I./memory \
    -I./bridge \
    -I./ckpt \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include


LIB_PATHS = \
    -L$(TENSOR_LIB_DIR)/lib \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib


LIBS = \
	-lnccl \
	-lcudart \
	-lcublas \
	-lcurand \
	-lgomp \
	-lstdc++


SRCS = \
    main.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
    ckpt/ckpt.cpp

OBJS = $(SRCS:.cpp=.o)



.PHONY: all clean test test_mlp_forward


all: $(TARGET)


test: test_mlp_forward
	@echo "\n[TEST BUILD COMPLETE]"
	@echo "Run the test with: mpirun -np 2 ./test_mlp_forward"
	@echo ""


$(TARGET): $(OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $(TARGET) $(OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Build complete."


TEST_MLP_SRCS = \
	test_mlp_forward.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_MLP_OBJS = $(TEST_MLP_SRCS:.cpp=.o)

test_mlp_forward: $(TEST_MLP_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(TEST_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."


%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

benchmarks/%.o: benchmarks/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

tests/%.o: tests/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# CUDA compilation rule for .cu files
tensor/%.o: tensor/%.cu
	@echo "[NVCC COMPILE] $<"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -c $< -o $@ $(INCLUDES)

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the 'Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi


clean:
	@echo "[CLEAN] Removing object files and executables..."
	rm -f $(OBJS) $(TARGET) $(TEST_MLP_OBJS) test_mlp_forward
	rm -f tests/*.o tests/test_device_mesh tests/test_matmul
	rm -f benchmarks/*.o benchmarks/nccl_benchmark benchmarks/matmul_benchmark benchmarks/row_parallel_breakdown benchmarks/row_parallel_breakdown_custom
	rm -f tensor/dtensor_custom.o bridge/tensor_ops_bridge_custom.o



.PHONY: test_device_mesh nccl_benchmark test_matmul matmul_benchmark row_parallel_breakdown row_parallel_breakdown_custom gradient_sync_example

TEST_DEVICE_MESH_SRCS = \
	tests/test_device_mesh.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_DEVICE_MESH_OBJS = $(TEST_DEVICE_MESH_SRCS:.cpp=.o)

test_device_mesh: $(TEST_DEVICE_MESH_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_DEVICE_MESH_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."
	@echo "\nRun : mpirun -np 2 ./tests/test_device_mesh\n"

NCCL_BENCHMARK_SRCS = benchmarks/nccl_benchmark_1.cpp
NCCL_BENCHMARK_OBJS = $(NCCL_BENCHMARK_SRCS:.cpp=.o)

nccl_benchmark: $(NCCL_BENCHMARK_OBJS)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(NCCL_BENCHMARK_OBJS) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Benchmark build complete."
	@echo "\nRun : mpirun -np 2 ./benchmarks/nccl_benchmark\n"

MATMUL_BENCHMARK_SRCS = \
	benchmarks/matmul_benchmark.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

MATMUL_BENCHMARK_OBJS = $(MATMUL_BENCHMARK_SRCS:.cpp=.o)

matmul_benchmark: $(MATMUL_BENCHMARK_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(MATMUL_BENCHMARK_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MatMul Benchmark build complete."
	@echo "\nRun : mpirun -np 2 ./benchmarks/matmul_benchmark\n"


ROW_BREAKDOWN_SRCS = \
	benchmarks/row_parallel_breakdown.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

ROW_BREAKDOWN_OBJS = $(ROW_BREAKDOWN_SRCS:.cpp=.o)

row_parallel_breakdown: $(ROW_BREAKDOWN_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(ROW_BREAKDOWN_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Row-Parallel Breakdown build complete."
	@echo "\nRun : mpirun -np 2 ./benchmarks/row_parallel_breakdown\n"


TEST_MATMUL_SRCS = \
	tests/test_matmul.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_MATMUL_OBJS = $(TEST_MATMUL_SRCS:.cpp=.o)

test_matmul: $(TEST_MATMUL_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_MATMUL_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MatMul test build complete."
	@echo "\nRun : mpirun -np 2 ./tests/test_matmul\n"

ROW_BREAKDOWN_CUSTOM_SRCS = \
	benchmarks/row_parallel_breakdown.cpp \
	tensor/dtensor_own.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge_own.cpp \
	ckpt/ckpt.cpp

ROW_BREAKDOWN_CUSTOM_OBJS = $(ROW_BREAKDOWN_CUSTOM_SRCS:.cpp=.o)

row_parallel_breakdown_custom: $(ROW_BREAKDOWN_CUSTOM_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating custom benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(ROW_BREAKDOWN_CUSTOM_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Custom Row-Parallel Breakdown build complete."
	@echo "\nRun : mpirun -np 2 ./benchmarks/row_parallel_breakdown_custom\n"

GRADIENT_SYNC_SRCS = \
	examples/gradient_sync_example.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

GRADIENT_SYNC_OBJS = $(GRADIENT_SYNC_SRCS:.cpp=.o)

examples/%.o: examples/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

gradient_sync_example: $(GRADIENT_SYNC_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o examples/$@ $(GRADIENT_SYNC_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Gradient Sync Example build complete."
	@echo "\nRun with 4 GPUs: mpirun -np 4 ./examples/gradient_sync_example\n"

# Tensor Parallel MLP Example
TP_MLP_SRCS = \
	examples/tensor_parallel_mlp.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TP_MLP_OBJS = $(TP_MLP_SRCS:.cpp=.o)

tensor_parallel_mlp: $(TP_MLP_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o examples/$@ $(TP_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Tensor Parallel MLP Example build complete."
	@echo "\nRun with 2 GPUs: mpirun -np 2 ./examples/tensor_parallel_mlp\n"

# MLP Benchmark Test
TEST_MLP_BENCHMARK_SRCS = \
	tests/test_mlp_benchmark.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_MLP_BENCHMARK_OBJS = $(TEST_MLP_BENCHMARK_SRCS:.cpp=.o)
TEST_MLP_BENCHMARK_OBJS := $(TEST_MLP_BENCHMARK_OBJS:.cu=.o)

.PHONY: test_mlp_benchmark

test_mlp_benchmark: $(TEST_MLP_BENCHMARK_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_MLP_BENCHMARK_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MLP Benchmark build complete."
	@echo "\nRun single GPU: ./tests/test_mlp_benchmark"
	@echo "Run 2 GPUs: mpirun -np 2 --allow-run-as-root ./tests/test_mlp_benchmark\n"

# DTensor Factory Functions Test
TEST_DTENSOR_FACTORY_SRCS = \
	tests/test_dtensor_factories.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_DTENSOR_FACTORY_OBJS = $(TEST_DTENSOR_FACTORY_SRCS:.cpp=.o)
TEST_DTENSOR_FACTORY_OBJS := $(TEST_DTENSOR_FACTORY_OBJS:.cu=.o)

.PHONY: test_dtensor_factories

test_dtensor_factories: $(TEST_DTENSOR_FACTORY_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_DTENSOR_FACTORY_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] DTensor Factory test build complete."
	@echo "\nRun: mpirun -np 2 --allow-run-as-root ./tests/test_dtensor_factories\n"

# DTensor Creation Methods Demo Test
TEST_DTENSOR_CREATION_SRCS = \
	tests/test_dtensor_creation_methods.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_DTENSOR_CREATION_OBJS = $(TEST_DTENSOR_CREATION_SRCS:.cpp=.o)
TEST_DTENSOR_CREATION_OBJS := $(TEST_DTENSOR_CREATION_OBJS:.cu=.o)

.PHONY: test_dtensor_creation_methods

test_dtensor_creation_methods: $(TEST_DTENSOR_CREATION_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_DTENSOR_CREATION_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] DTensor Creation Methods test build complete."
	@echo "\nRun: mpirun -np 2 --allow-run-as-root ./tests/test_dtensor_creation_methods\n"