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

# --- Target Executables ---
TARGET = dtensor_main
TEST_MLP_TARGET = test_mlp_forward

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
    -Xlinker -lmpi \
    -Xlinker -lmpi_cxx

# --- Source & Object Files ---
# Core sources needed by all executables
CORE_SRCS = \
    tensor/dtensor.cpp \
    process_group/process_group_nccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
    ckpt/ckpt.cpp

# Main executable sources
MAIN_SRC = main.cpp
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

# MLP test sources
TEST_MLP_SRC = test_mlp_forward.cpp
TEST_MLP_OBJ = $(TEST_MLP_SRC:.cpp=.o)

# All object files
CORE_OBJS = $(CORE_SRCS:.cpp=.o)
OBJS = $(MAIN_OBJ) $(CORE_OBJS)

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean test test_mlp run_test_mlp

# Default target
all: $(TARGET)

# Main test target now runs the MLP test
test: run_test_mlp

# 1. Linking Rule for Main Executable
$(TARGET): $(MAIN_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating main executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Build complete. Run with: mpirun -np 2 ./$(TARGET)"

# 2. Linking Rule for MLP Test Executable
test_mlp: $(TEST_MLP_TARGET)

$(TEST_MLP_TARGET): $(TEST_MLP_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $^ $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."

# 3. Rule to run the MLP test
run_test_mlp: $(TEST_MLP_TARGET)
	@echo "\n[RUNNING TEST] mpirun -np 2 ./$(TEST_MLP_TARGET)"
	mpirun -np 2 --allow-run-as-root ./$(TEST_MLP_TARGET)

# 4. Compilation Rule for C++ Source Files
# Uses mpic++ to compile .cpp -> .o
%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 5. Check for Static Library
$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the 'Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi

# 6. Cleanup
clean:
	@echo "[CLEAN] Removing object files and executables..."
	rm -f $(MAIN_OBJ) $(CORE_OBJS) $(TEST_MLP_OBJ) $(TARGET) $(TEST_MLP_TARGET)