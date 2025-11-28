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
# Core Source files (Core Distributed Tensor)
CORE_SRCS = \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/process_group_nccl.cpp \
    process_group/stream_pool.cpp \
    process_group/error_handler.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
    ckpt/ckpt.cpp

# Build directory
BUILD_DIR = build

# Main executable sources
MAIN_SRC = main.cpp

# Object files (in build directory)
CORE_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CORE_SRCS))
MAIN_OBJ = $(BUILD_DIR)/$(MAIN_SRC:.cpp=.o)

# All object files
OBJS = $(MAIN_OBJ) $(CORE_OBJS)

# Target executable path
TARGET = $(BUILD_DIR)/dtensor_main

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean test help
# === Utility Targets ===

.PHONY: all clean run help test

all: $(TARGET)

clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -rf $(BUILD_DIR)
	@echo "[CLEAN] Build directory cleaned"

run: $(TARGET)
	mpirun -np 2 --allow-run-as-root ./$(TARGET)

help:
	@echo "DTensor v2.0 Makefile"
	@echo "Targets:"
	@echo "  all    - Build dtensor_main (default)"
	@echo "  clean  - Remove build artifacts"
	@echo "  run    - Run with MPI (2 ranks)"
	@echo "  test         - Run all tests (delegates to tests/Makefile)"
	@echo "  clean        - Remove object files and executables"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "For test-specific targets, use:"

# Main test target delegates to tests directory
test:
	@echo "\n=========================================="
	@echo "  Running DTensor Test Suite"
	@echo "=========================================="
	cd tests && $(MAKE) run_all


# === Build Rules ===

# Create build directory structure
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)/{tensor,process_group,memory,bridge,ckpt}

# Main target
$(TARGET): $(BUILD_DIR) $(MAIN_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating main executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(MAIN_OBJ) $(CORE_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Build complete. Run with: mpirun -np 2 ./$(TARGET)"

# Compile main.cpp
$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	@echo "[COMPILE] $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Pattern rule for object files
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@echo "[COMPILE] $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Check for Static Library
$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the 'Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi

# 4. Cleanup
clean:
	@echo "[CLEAN] Removing object files and executables..."
	rm -f $(MAIN_OBJ) $(CORE_OBJS) $(TARGET)
	cd tests && $(MAKE) clean
	@echo "[CLEAN] Complete"