# =============================================================================
# Makefile for DTensor Tests
# Uses unparalleled.a library for fast compilation
# =============================================================================

# --- Parent Directory Variables ---
PARENT_DIR := ..
TENSOR_LIB_DIR := $(PARENT_DIR)/Tensor-Implementations
TENSOR_LIB_A := $(TENSOR_LIB_DIR)/lib/libtensor.a
UNPARALLELED_A := $(PARENT_DIR)/lib/unparalleled.a

# --- Compiler Configuration ---
CXX = mpic++
NVCC_LINKER = /usr/local/cuda/bin/nvcc

# --- Build Flags ---
CXXFLAGS = -std=c++17 -O3 -fPIC -g -DWITH_CUDA
NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

# --- Paths ---
CUDA_HOME := /usr/local/cuda

# --- Include Directories ---
INCLUDES = \
    -I$(PARENT_DIR) \
    -I$(PARENT_DIR)/include \
    -I$(PARENT_DIR)/tensor \
    -I$(PARENT_DIR)/process_group \
    -I$(PARENT_DIR)/memory \
    -I$(PARENT_DIR)/bridge \
    -I$(PARENT_DIR)/ckpt \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include

# --- Library Directories ---
LIB_PATHS = \
    -L$(PARENT_DIR)/lib \
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
    -lstdc++

# --- All test executables ---
TESTS = \
    test_public_headers \
    test_dtensor_factories \
    test_dtensor_creation_methods \
    test_mlp_benchmark \
    test_mlp_forward \
    test_redistribute \
    test_process_group_nccl \
    test_rotate3d_sharding \
    test_lazy_partial_matmul \
    test_partial_layout \
    test_attention_parallel \
    test_gpu_native_init \
    test_p2p_scatter \
    test_benchmark_init \
    test_pipelined_init \
    test_architecture_benchmark \
    test_benchmark_layouts \
    test_fault_tolerance

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean help check_lib

# Disable built-in rules
.SUFFIXES:

# Default target
all: check_lib $(TESTS)
	@echo "\n[SUCCESS] All tests built!"

# Check library exists
check_lib:
	@if [ ! -f $(UNPARALLELED_A) ]; then \
		echo "[INFO] Building library first..."; \
		cd $(PARENT_DIR) && $(MAKE) lib; \
	fi

# Help target
help:
	@echo "DTensor Test Suite Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make                  - Build all tests"
	@echo "  make test_name        - Build specific test"
	@echo "  make clean            - Remove all test executables"
	@echo ""
	@echo "Run example:"
	@echo "  make test_public_headers"
	@echo "  mpirun -np 2 ./test_public_headers"
	@echo ""
	@echo "Available tests:"
	@for t in $(TESTS); do echo "  $$t"; done

# =============================================================================
# Compilation Rule
# =============================================================================

%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# =============================================================================
# Linking Rule (explicit for each test)
# =============================================================================

define TEST_RULE
$(1): $(1).o $(UNPARALLELED_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $$@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $$@ $$< $(UNPARALLELED_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $$@ built"
	@echo "Run: mpirun -np 2 ./$$@\n"
endef

$(foreach test,$(TESTS),$(eval $(call TEST_RULE,$(test))))

# =============================================================================
# Library Checks
# =============================================================================

$(UNPARALLELED_A):
	@echo "[INFO] Library not found, building..."
	cd $(PARENT_DIR) && $(MAKE) lib

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] TensorLib not found: $(TENSOR_LIB_A)"; \
		exit 1; \
	fi

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "[CLEAN] Removing test executables and objects..."
	rm -f $(TESTS) *.o
	@echo "[CLEAN] Done"
