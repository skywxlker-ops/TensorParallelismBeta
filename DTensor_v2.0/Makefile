# ==========================================================
# DTensor_v2.2 Build System (Phase 4 - N-D Tensor Integration + cuBLAS)
# ==========================================================

CXX := mpicxx
NVCC := nvcc

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
ROOT := .
SRCDIR := $(ROOT)
OBJDIR := $(ROOT)/lib/objects
BINDIR := $(ROOT)
TENSOR_PATH := $(ROOT)/Tensor-Implementations
BRIDGE_PATH := $(ROOT)/bridge

# ----------------------------------------------------------
# Targets
# ----------------------------------------------------------
TENSOR_LIB_SO := $(TENSOR_PATH)/lib/libtensor.so
DTENSOR_LIB_SO := $(ROOT)/lib/libdtensor.so
TARGET := $(BINDIR)/dtensor_main

# ----------------------------------------------------------
# CUDA Configuration
# ----------------------------------------------------------
CUDA_PATH := /usr/local/cuda

# ----------------------------------------------------------
# Include Paths
# ----------------------------------------------------------
INCLUDES := \
    -I$(TENSOR_PATH)/include \
    -I$(TENSOR_PATH)/include/core \
    -I$(TENSOR_PATH)/include/device \
    -I$(TENSOR_PATH)/include/ops \
    -I$(TENSOR_PATH)/include/dtype \
    -I/usr/lib/x86_64-linux-gnu/openmpi/include \
    -I$(CUDA_PATH)/include \
    -I$(SRCDIR) \
    -I$(SRCDIR)/process_group \
    -I$(SRCDIR)/memory \
    -I$(SRCDIR)/tensor \
    -I$(SRCDIR)/ckpt \
    -I$(BRIDGE_PATH)

# ----------------------------------------------------------
# Compiler & Linker Flags
# ----------------------------------------------------------
CXXFLAGS := -std=c++20 -O3 -fPIC -Wall -Wextra -DWITH_CUDA -fopenmp -Wno-cast-function-type $(INCLUDES)
NVCCFLAGS := -std=c++20 -O3 -Xcompiler "-fPIC -fopenmp" -DWITH_CUDA $(INCLUDES)

LDFLAGS := \
    -L$(CUDA_PATH)/lib64 \
    -L$(TENSOR_PATH)/lib \
    -L$(ROOT)/lib \
    -lcudart -lnccl -lcublas -lmpi -lgomp -lpthread -lcurand \
    -Wl,-rpath=$(CUDA_PATH)/lib64:$(TENSOR_PATH)/lib:$(ROOT)/lib

# ----------------------------------------------------------
# Source Discovery
# ----------------------------------------------------------
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp' ! -path "*/Tests/*")
CU_SOURCES  := $(shell find $(SRCDIR) -name '*.cu' 2>/dev/null)

OBJECTS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES)) \
           $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES))

MAIN_SRC := $(shell find $(SRCDIR) -maxdepth 2 -name "main.cpp" 2>/dev/null)
MAIN_OBJ := $(OBJDIR)/$(MAIN_SRC:.cpp=.o)

# ==========================================================
# Build Rules
# ==========================================================

all: prepare $(TENSOR_LIB_SO) $(DTENSOR_LIB_SO) $(TARGET)

prepare:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(TENSOR_PATH)/lib
	@mkdir -p $(ROOT)/lib
	@echo "==[Prepare]== Directories ready =="

# ----------------------------------------------------------
# TensorLib (dependency)
# ----------------------------------------------------------
$(TENSOR_LIB_SO):
	@echo "==[TensorLib]== Building TensorLib shared library =="
	$(MAKE) -C $(TENSOR_PATH)

# ----------------------------------------------------------
# Object Compilation
# ----------------------------------------------------------
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ----------------------------------------------------------
# DTensor Shared Library
# ----------------------------------------------------------
$(DTENSOR_LIB_SO): $(filter-out $(MAIN_OBJ),$(OBJECTS)) $(TENSOR_LIB_SO)
	@echo "==[DTensor]== Linking shared library =="
	$(CXX) -shared $(CXXFLAGS) $(filter-out $(MAIN_OBJ),$(OBJECTS)) -ltensor -o $@ $(LDFLAGS)

# ----------------------------------------------------------
# Executable
# ----------------------------------------------------------
$(TARGET): $(DTENSOR_LIB_SO) $(MAIN_OBJ)
	@echo "==[DTensor]== Linking main executable =="
	$(CXX) $(CXXFLAGS) $(MAIN_OBJ) -L$(TENSOR_PATH)/lib -ltensor -Llib -ldtensor -o $@ $(LDFLAGS)
	@echo "\n✅ Build complete: $(TARGET)\n"

# ----------------------------------------------------------
# Run Options
# ----------------------------------------------------------
run: $(TARGET)
	mpirun -np 2 ./dtensor_main

run-save: $(TARGET)
	mpirun -np 2 ./dtensor_main --save

run-load: $(TARGET)
	mpirun -np 2 ./dtensor_main --load

# Simple 3D tensor test (2×2×2)
run-3d: $(TARGET)
	mpirun -np 2 ./dtensor_main --3d

# Custom N-D tensor shape (example: make run-shape SHAPE=2x3x3)
run-shape: $(TARGET)
	@if [ -z "$(SHAPE)" ]; then \
		echo "Usage: make run-shape SHAPE=2x3x3"; \
	else \
		mpirun -np 2 ./dtensor_main --shape=$(SHAPE); \
	fi

# 1D vector (legacy)
run-len: $(TARGET)
	@if [ -z "$(LEN)" ]; then \
		echo "Usage: make run-len LEN=1024"; \
	else \
		mpirun -np 2 ./dtensor_main --len=$(LEN); \
	fi

# ----------------------------------------------------------
# Cleanup
# ----------------------------------------------------------
clean:
	@echo "==[Clean]== Removing build artifacts =="
	rm -rf $(OBJDIR) $(DTENSOR_LIB_SO) $(TARGET)
	$(MAKE) -C $(TENSOR_PATH) clean || true

.PHONY: all clean prepare run run-save run-load run-len run-3d run-shape
