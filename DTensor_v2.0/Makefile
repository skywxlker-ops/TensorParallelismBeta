# =============================================================================
# DTensor v2.0 - Backend Makefile
# Builds: unparalleled.a (static) and unparalleled.so (shared)
# =============================================================================

CXX = mpic++
NVCC_LINKER = /usr/local/cuda/bin/nvcc

CXXFLAGS = -std=c++17 -O3 -fPIC -g -DWITH_CUDA
NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := ./Tensor-Implementations
TENSOR_LIB_A   := $(TENSOR_LIB_DIR)/lib/libtensor.a

# --- Include Directories ---
INCLUDES = \
    -I. \
    -I./include \
    -I./tensor \
    -I./process_group \
    -I./memory \
    -I./bridge \
    -I./ckpt \
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
    -lstdc++

# =============================================================================
# UNPARALLELED LIBRARY
# =============================================================================
LIBDIR := lib
LIB_A  := $(LIBDIR)/unparalleled.a
LIB_SO := $(LIBDIR)/unparalleled.so

# Core library sources
LIB_SRCS := \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    tensor/placement.cpp \
    tensor/fused_shard_kernel.cu \
    tensor/distributed_loss_kernels.cu \
    process_group/processGroupNccl.cpp \
    process_group/reverse.cu \
    bridge/bridge.cpp \
    nn/nn.cpp \
    nn/CustomDNN.cpp

# Object files go into lib/objects/
LIB_OBJDIR := $(LIBDIR)/objects
LIB_OBJS := $(patsubst %.cpp,$(LIB_OBJDIR)/%.o,$(filter %.cpp,$(LIB_SRCS)))
LIB_OBJS += $(patsubst %.cu,$(LIB_OBJDIR)/%.o,$(filter %.cu,$(LIB_SRCS)))

# =============================================================================
# BUILD TARGETS
# =============================================================================

.PHONY: all lib lib-static lib-shared clean help

# Default: build both libraries
all: lib

# Build both static and shared libraries
lib: $(LIB_A) $(LIB_SO)
	@echo "\n[SUCCESS] Both libraries built:"
	@echo "  - $(LIB_A)"
	@echo "  - $(LIB_SO)"

lib-static: $(LIB_A)
lib-shared: $(LIB_SO)

# Help
help:
	@echo "DTensor v2.0 - Backend Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make lib         - Build unparalleled.a and unparalleled.so"
	@echo "  make lib-static  - Build only static library"
	@echo "  make lib-shared  - Build only shared library"
	@echo "  make clean       - Remove all build artifacts"
	@echo ""
	@echo "For tests: cd tests && make"

# =============================================================================
# COMPILATION RULES
# =============================================================================

# C++ sources -> lib/objects/
$(LIB_OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# CUDA sources -> lib/objects/
$(LIB_OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "[NVCC COMPILE] $< -> $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -c $< -o $@ $(INCLUDES)

# =============================================================================
# LIBRARY BUILD RULES
# =============================================================================

$(LIB_A): $(LIB_OBJS) $(TENSOR_LIB_A)
	@mkdir -p $(LIBDIR)
	@echo "\n[ARCHIVE] Creating static library: $@"
	@ar rcs $@ $(LIB_OBJS)
	@echo "[SUCCESS] Static library created: $@"

# Shared library
$(LIB_SO): $(LIB_OBJS) $(TENSOR_LIB_A)
	@mkdir -p $(LIBDIR)
	@echo "\n[LINKING] Creating shared library: $@"
	$(NVCC_LINKER) -shared $(NVCC_LINK_FLAGS) $(LIB_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS) -o $@
	@echo "[SUCCESS] Shared library created: $@"

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] TensorLib not found: $(TENSOR_LIB_A)"; \
		echo "        Run: cd Tensor-Implementations && make\n"; \
		exit 1; \
	fi

# =============================================================================
# BENCHMARKS
# =============================================================================

.PHONY: shard_benchmarking

shard_benchmarking: $(LIB_A) $(TENSOR_LIB_A)
	@mkdir -p benchmarks
	@echo "\n[COMPILE] benchmarks/shard_benchmarking.cpp"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c benchmarks/shard_benchmarking.cpp -o benchmarks/shard_benchmarking.o
	@echo "[LINKING] shard_benchmarking"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/shard_benchmarking benchmarks/shard_benchmarking.o $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Benchmark built: benchmarks/shard_benchmarking"
	@echo "\nRun: mpirun -np 2 --allow-run-as-root ./benchmarks/shard_benchmarking\n"

shard_benchmarking2: $(LIB_A) $(TENSOR_LIB_A)
	@mkdir -p benchmarks
	@echo "\n[COMPILE] benchmarks/shard_benchmarking2.cpp"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c benchmarks/shard_benchmarking2.cpp -o benchmarks/shard_benchmarking2.o
	@echo "[LINKING] shard_benchmarking2"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/shard_benchmarking2 benchmarks/shard_benchmarking2.o $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Benchmark built: benchmarks/shard_benchmarking2"
	@echo "\nRun: mpirun -np 2 --allow-run-as-root ./benchmarks/shard_benchmarking2\n"

shard_benchmarking_vary: $(LIB_A) $(TENSOR_LIB_A)
	@mkdir -p benchmarks
	@echo "\n[COMPILE] benchmarks/shard_benchmarking_vary.cpp"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c benchmarks/shard_benchmarking_vary.cpp -o benchmarks/shard_benchmarking_vary.o
	@echo "[LINKING] shard_benchmarking_vary"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/shard_benchmarking_vary benchmarks/shard_benchmarking_vary.o $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Benchmark built: benchmarks/shard_benchmarking_vary"
	@echo "\nRun: mpirun -np 2 --allow-run-as-root ./benchmarks/shard_benchmarking_vary\n"

DMLP_EXAMPLE_SRCS = \
    examples/dmlp_example.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl_new.cpp \
    bridge/tensor_ops_bridge.cpp \
	tensor/placement.cpp

DMLP_EXAMPLE_OBJS = $(DMLP_EXAMPLE_SRCS:.cpp=.o) $(CUDA_OBJS)

dmlp_example: $(DMLP_EXAMPLE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(DMLP_EXAMPLE_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DModule MLP Example build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/dmlp_example\n"

# =============================================================================
# EXAMPLES
# =============================================================================

hooks_example: $(LIB_A) $(TENSOR_LIB_A)
	@mkdir -p nn/examples
	@echo "\n[COMPILE] nn/examples/hooks_example.cpp"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c nn/examples/hooks_example.cpp -o nn/examples/hooks_example.o
	@echo "[LINKING] hooks_example"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o nn/examples/hooks_example nn/examples/hooks_example.o $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Example built: nn/examples/hooks_example"
	@echo "\nRun: mpirun -np 2 ./nn/examples/hooks_example\n"

gpt2_test: $(LIB_A) $(TENSOR_LIB_A)
	@mkdir -p script/script11
	@echo "\n[COMPILE] script/script11/gpt2_test.cpp"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c script/script11/gpt2_test.cpp -o script/script11/gpt2_test.o
	@echo "[LINKING] gpt2_test"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o script/script11/gpt2_test script/script11/gpt2_test.o $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] GPT-2 Test built: script/script11/gpt2_test"
	@echo "\nRun: mpirun -np 2 --allow-run-as-root ./script/script11/gpt2_test\n"

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -rf $(LIB_OBJDIR) $(LIB_A) $(LIB_SO) benchmarks/*.o benchmarks/shard_benchmarking benchmarks/shard_benchmarking2 benchmarks/shard_benchmarking_vary nn/examples/*.o nn/examples/hooks_example
	@echo "[CLEAN] Done"