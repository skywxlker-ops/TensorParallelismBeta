# Makefile for dtensor_full_collectives

# CUDA and NCCL paths
CUDA_HOME ?= /usr/local/cuda-13.0
NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_LIB := $(CUDA_HOME)/lib64
CUDA_INC := $(CUDA_HOME)/include

# Compiler flags
CXXFLAGS := -std=c++17
LDFLAGS := -lnccl -lcudart

# Target
TARGET := dtensor_full_collectives
SRC := dtensor_full_collectives.cpp

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	@echo "Compiling $< ..."
	CUDA_HOME=$(CUDA_HOME) \
	PATH=$(CUDA_HOME)/bin:$$PATH \
	LD_LIBRARY_PATH=$(CUDA_LIB):$$LD_LIBRARY_PATH \
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Run target
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET)
