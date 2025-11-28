# Makefile for RAII Examples

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -O2
NVCCFLAGS = -std=c++17 -O2
CUDA_PATH = /usr/local/cuda
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

BIN_DIR = bin
OBJ_DIR = obj

# Create directories if they don't exist
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

# Targets
TARGETS = $(BIN_DIR)/simple_array \
          $(BIN_DIR)/cpu_matmul \
          $(BIN_DIR)/gpu_matmul \
          $(BIN_DIR)/benchmark

.PHONY: all clean

all: $(TARGETS)

# Simple array example
$(BIN_DIR)/simple_array: simple_array.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Benchmark
$(BIN_DIR)/benchmark: benchmark.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# CPU matmul
$(BIN_DIR)/cpu_matmul: cpu_matmul.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# GPU matmul kernel
$(OBJ_DIR)/matmul_kernel.o: matmul_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# GPU matmul
$(BIN_DIR)/gpu_matmul: gpu_matmul.cpp $(OBJ_DIR)/matmul_kernel.o
	$(NVCC) $(NVCCFLAGS) -I. $^ -o $@

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

help:
	@echo "RAII Examples Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build all examples"
	@echo "  simple_array  - Build simple array example"
	@echo "  benchmark     - Build benchmark"
	@echo "  cpu_matmul    - Build CPU matmul"
	@echo "  gpu_matmul    - Build GPU matmul (requires CUDA)"
	@echo "  clean         - Remove all built files"
	@echo ""
	@echo "Usage:"
	@echo "  make all"
	@echo "  ./bin/simple_array"
	@echo "  ./bin/benchmark"

# Convenience targets
simple_array: $(BIN_DIR)/simple_array
benchmark: $(BIN_DIR)/benchmark
cpu_matmul: $(BIN_DIR)/cpu_matmul
gpu_matmul: $(BIN_DIR)/gpu_matmul
