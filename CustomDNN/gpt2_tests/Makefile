# =============================================================================
# CustomDNN GPT-2 Tests Makefile
# Uses customdnn.a library for fast compilation
# =============================================================================

# --- Parent Directory Variables ---
PARENT_DIR := ..
DTENSOR_DIR := ../../DTensor
TENSOR_LIB_DIR := $(DTENSOR_DIR)/Tensor-Implementations
TENSOR_LIB_A := $(TENSOR_LIB_DIR)/lib/libtensor.a
CUSTOMDNN_A := $(PARENT_DIR)/lib/customdnn.a

# --- Compiler Configuration ---
CXX = /usr/bin/mpic++
NVCC_LINKER = /usr/local/cuda-12.6/bin/nvcc

# --- Build Flags ---
CXXFLAGS = -std=c++17 -O3 -fPIC -g -DWITH_CUDA
NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -arch=sm_86 -ccbin=$(CXX)

# --- Paths ---
CUDA_HOME := /usr/local/cuda-12.6

# --- Include Directories ---
INCLUDES = \
    -I$(PARENT_DIR) \
    -I$(DTENSOR_DIR) \
    -I$(DTENSOR_DIR)/tensor \
    -I$(DTENSOR_DIR)/process_group \
    -I$(DTENSOR_DIR)/dnn \
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
    -lmpi \
    -lnccl \
    -lcudart \
    -lcublas \
    -lcurand \
    -lnvToolsExt \
    -lgomp \
    -lstdc++ \
    -lz -lpthread -ldl

LDFLAGS = -L/home/blu-bridge005/.local/lib -Xlinker -rpath -Xlinker /home/blu-bridge005/.local/lib

# --- All test executables ---
TESTS = gpt2_test gpt2_tp_test test

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean help check_lib

.SUFFIXES:

all: check_lib $(TESTS)
	@echo "\n[SUCCESS] All tests built!"

# Check library exists -- auto-build if missing
check_lib:
	@if [ ! -f $(CUSTOMDNN_A) ]; then \
		echo "[INFO] Building CustomDNN library first..."; \
		cd $(PARENT_DIR) && $(MAKE) lib; \
	fi

help:
	@echo "CustomDNN GPT-2 Test Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build all tests"
	@echo "  make gpt2_test    - Build GPT-2 test with CustomDNN"
	@echo "  make gpt2_tp_test - Build GPT-2 Tensor Parallel test with CustomDNN"
	@echo "  make clean        - Remove all test executables"
	@echo ""
	@echo "Run example:"
	@echo "  make gpt2_test"
	@echo "  mpirun -np 2 ./gpt2_test"
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
# Linking Rules
# =============================================================================

gpt2_test: gpt2_test.o $(CUSTOMDNN_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o $@ $< -Xlinker --start-group $(CUSTOMDNN_A) $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] $@ built"
	@echo "\nRun: mpirun -np 2 ./$@\n"

gpt2_tp_test: gpt2_tp_test.o $(CUSTOMDNN_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o $@ $< -Xlinker --start-group $(CUSTOMDNN_A) $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] $@ built"
	@echo "\nRun: mpirun -np 2 ./$@\n"

test: test.o $(CUSTOMDNN_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o $@ $< -Xlinker --start-group $(CUSTOMDNN_A) $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] $@ built"
	@echo "\nRun: mpirun -np 2 ./$@\n"

# =============================================================================
# Library Checks
# =============================================================================

$(CUSTOMDNN_A):
	@echo "[INFO] CustomDNN library not found, building..."
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
