# =============================================================================
# Makefile for MLP Testing Suite (Tensor Parallelism Profiling)
# =============================================================================

# --- Parent Directory Variables ---
PARENT_DIR := ../..
TESTS_DIR := ..
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
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include

# --- Library Directories ---
LIB_PATHS = \
    -L$(TENSOR_LIB_DIR)/lib \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib

# --- Libraries ---
LIBS = \
    -lnccl \
    -lcudart \
    -lcublas \
    -lcurand \
    -lgomp \
    -Xlinker -lmpi \
    -Xlinker -lmpi_cxx


# --- Core DTensor Objects (from parent directory) ---
CORE_SRCS = \
    $(PARENT_DIR)/tensor/dtensor.cpp \
    $(PARENT_DIR)/tensor/device_mesh.cpp \
    $(PARENT_DIR)/process_group/processGroupNccl.cpp \
    $(PARENT_DIR)/memory/cachingAllocator.cpp \
    $(PARENT_DIR)/bridge/tensor_ops_bridge.cpp \
    $(PARENT_DIR)/tensor/placement.cpp

CORE_CUDA_SRCS = \
    $(PARENT_DIR)/tensor/fused_shard_kernel.cu \
    $(PARENT_DIR)/process_group/reverse.cu

CORE_OBJS = $(CORE_SRCS:.cpp=.o) $(CORE_CUDA_SRCS:.cu=.o)

# --- Test Executables ---
TESTS = test_mlp_interconnect_bandwidth test_mlp_kernel_vs_sync test_mlp_async_timing test_mlp_arithmetic_intensity

# --- Test Object Files ---
TEST_BW_OBJ = test_mlp_interconnect_bandwidth.o
TEST_KERNEL_OBJ = test_mlp_kernel_vs_sync.o
TEST_ASYNC_OBJ = test_mlp_async_timing.o
TEST_AI_OBJ = test_mlp_arithmetic_intensity.o

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean help run_bandwidth run_kernel run_intensity run_all

# Default target - build all tests
all: $(TESTS)

help:
	@echo "MLP Testing Suite Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all tests (default)"
	@echo "  run_all          - Run all tests sequentially (TP=2)"
	@echo "  run_bandwidth    - Run interconnect bandwidth test"
	@echo "  run_kernel       - Run kernel vs sync time test"
	@echo "  run_intensity    - Run arithmetic intensity test"
	@echo "  clean            - Remove all test executables and objects"
	@echo ""
	@echo "Individual test targets:"
	@echo "  test_mlp_interconnect_bandwidth"
	@echo "  test_mlp_kernel_vs_sync"
	@echo "  test_mlp_arithmetic_intensity"

# --- Build Test Executables ---

test_mlp_interconnect_bandwidth: $(TEST_BW_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_mlp_kernel_vs_sync: $(TEST_KERNEL_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_mlp_async_timing: $(TEST_ASYNC_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

test_mlp_arithmetic_intensity: $(TEST_AI_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ build complete"

# --- Run Test Rules ---

run_bandwidth: test_mlp_interconnect_bandwidth
	@echo "\n=========================================="
	@echo "Running: MLP Interconnect Bandwidth Test (TP=2)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_mlp_interconnect_bandwidth

run_kernel: test_mlp_kernel_vs_sync
	@echo "\n=========================================="
	@echo "Running: MLP Kernel vs Sync Time Test (TP=2)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_mlp_kernel_vs_sync

run_intensity: test_mlp_arithmetic_intensity
	@echo "\n=========================================="
	@echo "Running: MLP Arithmetic Intensity Test (TP=2)"
	@echo "=========================================="
	mpirun -np 2 --allow-run-as-root ./test_mlp_arithmetic_intensity

run_all: run_bandwidth run_kernel run_intensity
	@echo "\n=========================================="
	@echo "  ALL MLP PROFILING TESTS COMPLETED!"
	@echo "=========================================="

# --- Compilation Rules ---

%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Core objects are in parent directory
$(PARENT_DIR)/%.o: $(PARENT_DIR)/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(PARENT_DIR)/%.o: $(PARENT_DIR)/%.cu
	@echo "[COMPILE] $<"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(INCLUDES) -c $< -o $@

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
	rm -f $(TESTS) *.o
	@echo "[CLEAN] Test directory cleaned"
