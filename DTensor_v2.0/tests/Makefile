# =============================================================================
# Makefile for DTensor Tests
# =============================================================================

# --- Parent Directory Variables ---
PARENT_DIR := ..
TENSOR_LIB_DIR := $(PARENT_DIR)/Tensor-Implementations
TENSOR_LIB_A := $(TENSOR_LIB_DIR)/lib/libtensor.a

# --- Compiler Configuration ---
CXX = mpic++
NVCC_LINKER = /usr/local/cuda/bin/nvcc

# --- Build Flags ---
CXXFLAGS = -std=c++17 -O3 -fPIC -g
NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

# --- Paths ---
CUDA_HOME := /usr/local/cuda

# --- Include Directories ---
INCLUDES = \
    -I$(PARENT_DIR) \
    -I$(PARENT_DIR)/tensor \
    -I$(PARENT_DIR)/process_group \
    -I$(PARENT_DIR)/memory \
    -I$(PARENT_DIR)/bridge \
    -I$(PARENT_DIR)/ckpt \
    -I$(PARENT_DIR)/cgadimpl/cgadimpl/include \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include

# --- Library Directories ---
LIB_PATHS = \
    -L$(TENSOR_LIB_DIR)/lib \
    -L$(PARENT_DIR)/cgadimpl/cgadimpl/build \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib

# --- Libraries ---
LIBS = \
    -lnccl \
    -lcudart \
    -lcublas \
    -lcurand \
    -Xlinker -lmpi \
    -Xlinker -lmpi_cxx
# Removed: -lcgad (autograd library not needed)

# --- Core DTensor Objects (from parent directory) ---
CORE_SRCS = \
    $(PARENT_DIR)/tensor/dtensor.cpp \
    $(PARENT_DIR)/tensor/device_mesh.cpp \
    $(PARENT_DIR)/process_group/processGroupNccl.cpp \
    $(PARENT_DIR)/memory/cachingAllocator.cpp \
    $(PARENT_DIR)/bridge/tensor_ops_bridge.cpp
# Note: dtensor_native.cpp removed (uses old ProcessGroup API), ckpt.cpp removed (directory doesn't exist)

CORE_OBJS = $(CORE_SRCS:.cpp=.o)
CORE_OBJS := $(CORE_OBJS:.cu=.o)


# --- Test Executables ---
TESTS = test_process_group test_process_group_nccl test_mlp_forward test_redistribute test_partial_layout test_fault_tolerance test_autograd_tp test_gpu_native_init test_p2p_scatter test_benchmark_init test_pipelined_init test_architecture_benchmark test_benchmark_layouts test_attention_parallel test_mlp_benchmark

# --- Test Source Files ---
TEST_STREAM_SRC = test_stream_pool.cpp
TEST_PG_SRC = test_process_group.cpp
TEST_MLP_SRC = test_mlp_forward.cpp
TEST_REDIST_SRC = test_redistribute.cpp

# --- Test Object Files ---
TEST_STREAM_OBJ = $(TEST_STREAM_SRC:.cpp=.o)
TEST_PG_OBJ = $(TEST_PG_SRC:.cpp=.o)
TEST_MLP_OBJ = $(TEST_MLP_SRC:.cpp=.o)
TEST_REDIST_OBJ = $(TEST_REDIST_SRC:.cpp=.o)
TEST_FAULT_SRC = test_fault_tolerance.cpp
TEST_FAULT_OBJ = $(TEST_FAULT_SRC:.cpp=.o)
TEST_AUTOGRAD_SRC = test_autograd_tp.cpp
TEST_AUTOGRAD_OBJ = $(TEST_AUTOGRAD_SRC:.cpp=.o)
TEST_GPU_INIT_SRC = test_gpu_native_init.cpp
TEST_GPU_INIT_OBJ = $(TEST_GPU_INIT_SRC:.cpp=.o)
TEST_P2P_SRC = test_p2p_scatter.cpp
TEST_P2P_OBJ = $(TEST_P2P_SRC:.cpp=.o)
TEST_BENCHMARK_SRC = test_benchmark_init.cpp
TEST_BENCHMARK_OBJ = $(TEST_BENCHMARK_SRC:.cpp=.o)
TEST_PIPELINED_SRC = test_pipelined_init.cpp
TEST_PIPELINED_OBJ = $(TEST_PIPELINED_SRC:.cpp=.o)
TEST_ARCH_SRC = test_architecture_benchmark.cpp
TEST_ARCH_OBJ = $(TEST_ARCH_SRC:.cpp=.o)
TEST_LAYOUTS_SRC = test_benchmark_layouts.cpp
TEST_LAYOUTS_OBJ = $(TEST_LAYOUTS_SRC:.cpp=.o)
TEST_ATTN_SRC = test_attention_parallel.cpp
TEST_ATTN_OBJ = $(TEST_ATTN_SRC:.cpp=.o)
TEST_MLP_BENCH_SRC = test_mlp_benchmark.cpp
TEST_MLP_BENCH_OBJ = $(TEST_MLP_BENCH_SRC:.cpp=.o)
TEST_PG_NCCL_SRC = test_process_group_nccl.cpp
TEST_PG_NCCL_OBJ = $(TEST_PG_NCCL_SRC:.cpp=.o)
TEST_PARTIAL_SRC = test_partial_layout.cpp
TEST_PARTIAL_OBJ = $(TEST_PARTIAL_SRC:.cpp=.o)
TEST_LAZY_PARTIAL_SRC = test_lazy_partial_matmul.cpp
TEST_LAZY_PARTIAL_OBJ = $(TEST_LAZY_PARTIAL_SRC:.cpp=.o)

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean run_all run_stream run_pg run_mlp run_redist run_fault run_gpu_init help

# Default target - build all tests
all: $(TESTS)

# Help target
help:
	@echo "DTensor Test Suite Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all tests (default)"
	@echo "  run_all      - Run all tests sequentially"
	@echo "  run_stream   - Run StreamPool tests (no MPI)"
	@echo "  run_pg       - Run ProcessGroup tests (2 ranks)"
	@echo "  run_mlp      - Run MLP forward pass test (2 ranks)"
	@echo "  run_redist   - Run redistribute test (2 ranks)"
	@echo "  run_fault    - Run fault tolerance tests (single process)"
	@echo "  run_gpu_init - Run GPU-native initialization test (2 ranks)"
	@echo "  clean        - Remove all test executables and objects"
	@echo ""
	@echo "Individual test targets:"
	@echo "  test_stream_pool"
	@echo "  test_process_group"
	@echo "  test_mlp_forward"
	@echo "  test_redistribute"
	@echo "  test_fault_tolerance"
	@echo "  test_gpu_native_init"

# --- Build Test Executables ---

test_stream_pool: $(TEST_STREAM_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_process_group: $(TEST_PG_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_mlp_forward: $(TEST_MLP_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_redistribute: $(TEST_REDIST_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_partial_layout: $(TEST_PARTIAL_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_lazy_partial_matmul: $(TEST_LAZY_PARTIAL_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_fault_tolerance: $(TEST_FAULT_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_autograd_tp: $(TEST_AUTOGRAD_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_gpu_native_init: $(TEST_GPU_INIT_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_p2p_scatter: $(TEST_P2P_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_benchmark_init: $(TEST_BENCHMARK_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_pipelined_init: $(TEST_PIPELINED_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_architecture_benchmark: $(TEST_ARCH_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_benchmark_layouts: $(TEST_LAYOUTS_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_attention_parallel: $(TEST_ATTN_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_mlp_benchmark: $(TEST_MLP_BENCH_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_process_group_nccl: $(TEST_PG_NCCL_OBJ) $(PARENT_DIR)/process_group/processGroupNccl.o $(PARENT_DIR)/tensor/dtensor.o $(PARENT_DIR)/tensor/device_mesh.o $(PARENT_DIR)/tensor/activation_kernels.o $(PARENT_DIR)/memory/cachingAllocator.o $(PARENT_DIR)/bridge/tensor_ops_bridge.o $(TENSOR_LIB_A)
	@echo "\\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

# --- Run Test Rules ---

run_stream: test_stream_pool
	@echo "\n=========================================="
	@echo "Running: StreamPool Tests (Single Process)"
	@echo "=========================================="
	./test_stream_pool

run_pg: test_process_group
	@echo "\n=========================================="
	@echo "Running: ProcessGroup Tests (2 Ranks)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_process_group

run_mlp: test_mlp_forward
	@echo "\n=========================================="
	@echo "Running: MLP Forward Pass Test (2 Ranks)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_mlp_forward

run_redist: test_redistribute
	@echo "\n=========================================="
	@echo "Running: Redistribute Test (2 Ranks)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_redistribute

run_fault: test_fault_tolerance
	@echo "\n=========================================="
	@echo "Running: Fault Tolerance Tests (Single Process)"
	@echo "=========================================="
	./test_fault_tolerance

run_gpu_init: test_gpu_native_init
	@echo "\n=========================================="
	@echo "Running: GPU-Native Init Test (2 Ranks)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_gpu_native_init

run_mlp_bench_single: test_mlp_benchmark
	@echo "\n=========================================="
	@echo "Running: MLP Benchmark (Single GPU)"
	@echo "=========================================="
	./test_mlp_benchmark

run_mlp_bench_tp: test_mlp_benchmark
	@echo "\n=========================================="
	@echo "Running: MLP Benchmark (2-GPU TP)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_mlp_benchmark

run_pg_nccl: test_process_group_nccl
	@echo "\n=========================================="
	@echo "Running: ProcessGroupNCCL Tests (2 Ranks)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_process_group_nccl

# Run all tests sequentially
run_all: run_stream run_pg run_mlp run_redist run_gpu_init
	@echo "\n=========================================="
	@echo "  ALL TESTS COMPLETED!"
	@echo "=========================================="

# --- Compilation Rules ---

%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Core objects are in parent directory
$(PARENT_DIR)/%.o: $(PARENT_DIR)/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) -I$(PARENT_DIR) -I$(PARENT_DIR)/tensor -I$(PARENT_DIR)/process_group -I$(PARENT_DIR)/memory -I$(PARENT_DIR)/bridge -I$(PARENT_DIR)/ckpt -I$(PARENT_DIR)/cgadimpl/cgadimpl/include -I$(TENSOR_LIB_DIR)/include -I$(CUDA_HOME)/include -c $< -o $@

# CUDA compilation rule for .cu files in parent directory
$(PARENT_DIR)/%.o: $(PARENT_DIR)/%.cu
	@echo "[NVCC COMPILE] $<"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -c $< -o $@ -I$(PARENT_DIR) -I$(PARENT_DIR)/tensor -I$(PARENT_DIR)/process_group -I$(PARENT_DIR)/memory -I$(PARENT_DIR)/bridge -I$(PARENT_DIR)/ckpt -I$(TENSOR_LIB_DIR)/include -I$(CUDA_HOME)/include

# --- Library Check ---

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the 'Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi

# --- Cleanup ---

clean:
	@echo "[CLEAN] Removing test executables and objects..."
	rm -f $(TESTS) $(TEST_STREAM_OBJ) $(TEST_PG_OBJ) $(TEST_MLP_OBJ) $(TEST_REDIST_OBJ) $(TEST_AUTOGRAD_OBJ) $(TEST_GPU_INIT_OBJ)
	@echo "[CLEAN] Test directory cleaned"

# Clean everything including core objects
clean_all: clean
	@echo "[CLEAN] Removing core objects from parent directory..."
	cd $(PARENT_DIR) && rm -f $(CORE_OBJS)
	@echo "[CLEAN] All objects cleaned"
