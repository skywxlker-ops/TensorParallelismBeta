# Makefile for building the C++ executable (dtensor_main)

# --- Compiler ---
# Use the MPI C++ compiler wrapper for C++ files
CXX = mpic++
# Use nvcc for the final link step
NVCC_LINKER = /usr/local/cuda/bin/nvcc

# --- Flags ---
CXXFLAGS = -std=c++17 -O3 -fPIC
# Flags for nvcc linker
# We tell nvcc to use mpic++ as its host compiler
NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

# --- Executable Name ---
TARGET = dtensor_main

# --- Paths ---
CUDA_HOME    := /usr/local/cuda
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
LIB_PATHS = \
    -L$(TENSOR_LIB_DIR)/lib \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib # May vary

# --- Libraries to Link (-l) ---
# We need to tell nvcc to pass the MPI libraries to the host linker (ld)
# using -Xlinker. libtensor.a must come after the objects.
LIBS = \
    -lnccl \
    -lcudart \
    -lcublas \
    -Xlinker -lmpi \
    -Xlinker -lmpi_cxx

# --- Source Files ---
SRCS = \
    main.cpp \
    tensor/dtensor.cpp \
    process_group/process_group_nccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
    ckpt/ckpt.cpp

# --- Object Files ---
OBJS = $(SRCS:.cpp=.o)

# --- Build Rules ---

# Default target: build the executable
all: $(TARGET)

# ==========================================================
# ✅ FIXED LINKING RULE
# ==========================================================
# Rule to link the executable
$(TARGET): $(OBJS) $(TENSOR_LIB_A)
	@echo "[LINKING WITH NVCC] $@"
	# Use NVCC to link everything
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $(TARGET) $(OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)

# Rule to compile .cpp files into .o files (using mpic++)
%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Ensure the static library exists
$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "Error: $(TENSOR_LIB_A) not found."; \
		echo "Please build Tensor-Implementations first."; \
		exit 1; \
	fi

# Clean up object files and the executable
clean:
	@echo "Cleaning up object files and executable..."
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean