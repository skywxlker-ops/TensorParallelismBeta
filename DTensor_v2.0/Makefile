# =============================================================================
# Makefile for DTensor Framework (Distributed Tensor Parallelism)
# =============================================================================

# --- Compiler Configuration ---
# Host compiler for C++ files (wraps g++ with MPI link flags)
CXX = mpic++

# Linker: We use nvcc to handle device code linking (libtensor.a contains CUDA kernels)
NVCC_LINKER = /usr/local/cuda/bin/nvcc

# --- Build Flags ---
# C++17 standard, Optimization level 3, Position Independent Code
CXXFLAGS = -std=c++17 -O3 -fPIC -g

# NVCC Linker Flags
# -ccbin=mpic++ tells nvcc to use the MPI wrapper as the host linker
NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

# --- Target Executable ---
TARGET = dtensor_main

# --- Paths (Adjust these if your system paths differ) ---
CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := ./Tensor-Implementations
TENSOR_LIB_A   := $(TENSOR_LIB_DIR)/lib/libtensor.a

# --- Include Directories (-I) ---
INCLUDES = \
    -I. \
    -I./tensor \
    -I./process_group \
    -I./memory \
    -I./bridge \
    -I./ckpt \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include

# --- Library Directories (-L) ---
# Note: OpenMPI path might vary (/usr/lib/x86_64-linux-gnu/openmpi/lib, etc.)
LIB_PATHS = \
    -L$(TENSOR_LIB_DIR)/lib \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib

# --- Libraries to Link (-l) ---
# -Xlinker passes flags directly to the host linker (ld)
LIBS = \
    -lnccl \
    -lcudart \
    -lcublas \
    -lcurand \
    -lgomp \
    -Xlinker -lmpi \
    -Xlinker -lmpi_cxx

# --- Source Files ---
SRCS = \
    main.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/process_group_nccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
    ckpt/ckpt.cpp

# --- Object Files ---
# Replaces .cpp with .o in the SRCS list
OBJS = $(SRCS:.cpp=.o)

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean test test_mlp_forward

# Default target
all: $(TARGET)

# Test target (builds and suggests how to run)
test: test_mlp_forward
	@echo "\n[TEST BUILD COMPLETE]"
	@echo "Run the test with: mpirun -np 2 ./test_mlp_forward"
	@echo ""

# 1. Linking Rule
# links object files + static tensor lib -> executable
$(TARGET): $(OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $(TARGET) $(OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Build complete."

# Test executable for MLP forward pass
TEST_MLP_SRCS = \
	test_mlp_forward.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/process_group_nccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_MLP_OBJS = $(TEST_MLP_SRCS:.cpp=.o)

test_mlp_forward: $(TEST_MLP_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(TEST_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."

# 2. Compilation Rule for C++ Source Files
# Uses mpic++ to compile .cpp -> .o
%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 3. Check for Static Library
$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the 'Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi

# 4. Cleanup
clean:
	@echo "[CLEAN] Removing object files and executables..."
	rm -f $(OBJS) $(TARGET) $(TEST_MLP_OBJS) test_mlp_forward
	rm -f tests/*.o tests/test_device_mesh tests/test_matmul
	rm -f benchmarks/*.o benchmarks/nccl_benchmark benchmarks/matmul_benchmark

# =============================================================================
# Additional Test and Benchmark Targets
# =============================================================================

.PHONY: test_device_mesh nccl_benchmark test_matmul matmul_benchmark

# Test for DeviceMesh functionality
TEST_DEVICE_MESH_SRCS = \
	tests/test_device_mesh.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/process_group_nccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_DEVICE_MESH_OBJS = $(TEST_DEVICE_MESH_SRCS:.cpp=.o)

test_device_mesh: $(TEST_DEVICE_MESH_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_DEVICE_MESH_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."
	@echo "\nRun with: mpirun -np 2 ./tests/test_device_mesh\n"

# NCCL Collectives Benchmark
NCCL_BENCHMARK_SRCS = benchmarks/nccl_benchmark.cpp
NCCL_BENCHMARK_OBJS = $(NCCL_BENCHMARK_SRCS:.cpp=.o)

nccl_benchmark: $(NCCL_BENCHMARK_OBJS)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(NCCL_BENCHMARK_OBJS) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Benchmark build complete."
	@echo "\nRun with: mpirun -np 2 ./benchmarks/nccl_benchmark\n"

# MatMul Benchmark
MATMUL_BENCHMARK_SRCS = \
	benchmarks/matmul_benchmark.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/process_group_nccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

MATMUL_BENCHMARK_OBJS = $(MATMUL_BENCHMARK_SRCS:.cpp=.o)

matmul_benchmark: $(MATMUL_BENCHMARK_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(MATMUL_BENCHMARK_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MatMul Benchmark build complete."
	@echo "\nRun with: mpirun -np 2 ./benchmarks/matmul_benchmark\n"

# Test for MatMul (Column-Parallel and Row-Parallel)
TEST_MATMUL_SRCS = \
	tests/test_matmul.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/process_group_nccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_MATMUL_OBJS = $(TEST_MATMUL_SRCS:.cpp=.o)

test_matmul: $(TEST_MATMUL_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $(TEST_MATMUL_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MatMul test build complete."
	@echo "\nRun with: mpirun -np 2 ./tests/test_matmul\n"