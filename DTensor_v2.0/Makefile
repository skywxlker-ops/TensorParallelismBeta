
CXX = mpic++


NVCC_LINKER = /usr/local/cuda/bin/nvcc

CXXFLAGS = -std=c++17 -O3 -fPIC -g -DWITH_CUDA


NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

TARGET = dtensor_main


CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := ./Tensor-Implementations
TENSOR_LIB_A   := $(TENSOR_LIB_DIR)/lib/libtensor.a


INCLUDES = \
    -I. \
    -I./tensor \
    -I./process_group \
    -I./memory \
    -I./bridge \
    -I./ckpt \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include


LIB_PATHS = \
    -L$(TENSOR_LIB_DIR)/lib \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib


LIBS = \
	-lnccl \
	-lcudart \
	-lcublas \
	-lcurand \
	-lgomp \
	-lstdc++


# ==================== UNPARALLELED LIBRARY ====================
LIBDIR := lib
LIB_A  := $(LIBDIR)/unparalleled.a
LIB_SO := $(LIBDIR)/unparalleled.so

# Core library sources (no main/test files)
LIB_SRCS := \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    process_group/reverse.cu \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp

# Object files go into lib/objects/
LIB_OBJDIR := $(LIBDIR)/objects
LIB_OBJS := $(patsubst %.cpp,$(LIB_OBJDIR)/%.o,$(filter %.cpp,$(LIB_SRCS)))
LIB_OBJS += $(patsubst %.cu,$(LIB_OBJDIR)/%.o,$(filter %.cu,$(LIB_SRCS)))


SRCS = \
    main.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp \
    ckpt/ckpt.cpp

OBJS = $(SRCS:.cpp=.o)



.PHONY: all clean test test_mlp_forward


all: $(TARGET)


test: test_mlp_forward
	@echo "\n[TEST BUILD COMPLETE]"
	@echo "Run the test with: mpirun -np 2 ./test_mlp_forward"
	@echo ""


$(TARGET): $(OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $(TARGET) $(OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Build complete."


TEST_MLP_SRCS = \
	test_mlp_forward.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge.cpp \
	ckpt/ckpt.cpp

TEST_MLP_OBJS = $(TEST_MLP_SRCS:.cpp=.o)

test_mlp_forward: $(TEST_MLP_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(TEST_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."


%.o: %.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

benchmarks/%.o: benchmarks/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

tests/%.o: tests/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# CUDA compilation rule for .cu files
tensor/%.o: tensor/%.cu
	@echo "[NVCC COMPILE] $<"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -c $< -o $@ $(INCLUDES)

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the 'Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi


clean:
	@echo "[CLEAN] Removing object files and executables..."
	rm -f $(OBJS) $(TARGET) $(TEST_MLP_OBJS) test_mlp_forward
	rm -rf $(LIB_OBJDIR) $(LIB_A) $(LIB_SO)
	rm -f tests/*.o tests/test_device_mesh tests/test_matmul
	rm -f benchmarks/*.o benchmarks/nccl_benchmark benchmarks/matmul_benchmark benchmarks/row_parallel_breakdown benchmarks/row_parallel_breakdown_custom
	rm -f tensor/dtensor_custom.o bridge/tensor_ops_bridge_custom.o


# ==================== LIBRARY BUILD TARGETS ====================
# Compilation rules for library objects (output to lib/objects/)
$(LIB_OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LIB_OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "[NVCC COMPILE] $< -> $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -c $< -o $@ $(INCLUDES)

# Static library
$(LIB_A): $(LIB_OBJS)
	@mkdir -p $(LIBDIR)
	@echo "\n[ARCHIVE] Creating static library: $@"
	ar rcs $@ $^
	@echo "[SUCCESS] Static library created: $@"

# Shared library
$(LIB_SO): $(LIB_OBJS) $(TENSOR_LIB_A)
	@mkdir -p $(LIBDIR)
	@echo "\n[LINKING] Creating shared library: $@"
	$(NVCC_LINKER) -shared $(NVCC_LINK_FLAGS) $(LIB_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS) -o $@
	@echo "[SUCCESS] Shared library created: $@"

# Stage 4: Convenience targets
.PHONY: lib lib-static lib-shared
lib: $(LIB_A) $(LIB_SO)
	@echo "\n[SUCCESS] Both libraries built:"
	@echo "  - $(LIB_A)"
	@echo "  - $(LIB_SO)"

lib-static: $(LIB_A)
lib-shared: $(LIB_SO)



.PHONY: test_device_mesh nccl_benchmark test_matmul matmul_benchmark row_parallel_breakdown row_parallel_breakdown_custom gradient_sync_example

# ==================== TESTS (link against unparalleled.a) ====================
# Output test objects to lib/objects/tests/
TEST_OBJDIR := $(LIB_OBJDIR)/tests
BENCHMARK_OBJDIR := $(LIB_OBJDIR)/benchmarks

# Compile test files to lib/objects/tests/
$(TEST_OBJDIR)/%.o: tests/%.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BENCHMARK_OBJDIR)/%.o: benchmarks/%.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test_device_mesh: $(TEST_OBJDIR)/test_device_mesh.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."
	@echo "\nRun: mpirun -np 2 ./tests/test_device_mesh\n"

nccl_benchmark: $(BENCHMARK_OBJDIR)/nccl_benchmark_1.o
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $< $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Benchmark build complete."
	@echo "\nRun: mpirun -np 2 ./benchmarks/nccl_benchmark\n"

matmul_benchmark: $(BENCHMARK_OBJDIR)/matmul_benchmark.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MatMul Benchmark build complete."
	@echo "\nRun: mpirun -np 2 ./benchmarks/matmul_benchmark\n"

row_parallel_breakdown: $(BENCHMARK_OBJDIR)/row_parallel_breakdown.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Row-Parallel Breakdown build complete."
	@echo "\nRun: mpirun -np 2 ./benchmarks/row_parallel_breakdown\n"

test_matmul: $(TEST_OBJDIR)/test_matmul.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MatMul test build complete."
	@echo "\nRun: mpirun -np 2 ./tests/test_matmul\n"

# Note: row_parallel_breakdown_custom uses custom dtensor_own.cpp - keep separate
ROW_BREAKDOWN_CUSTOM_SRCS = \
	benchmarks/row_parallel_breakdown.cpp \
	tensor/dtensor_own.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	memory/cachingAllocator.cpp \
	bridge/tensor_ops_bridge_own.cpp

ROW_BREAKDOWN_CUSTOM_OBJS = $(patsubst %.cpp,$(LIB_OBJDIR)/%.o,$(ROW_BREAKDOWN_CUSTOM_SRCS))

$(LIB_OBJDIR)/tensor/dtensor_own.o: tensor/dtensor_own.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LIB_OBJDIR)/bridge/tensor_ops_bridge_own.o: bridge/tensor_ops_bridge_own.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

row_parallel_breakdown_custom: $(ROW_BREAKDOWN_CUSTOM_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating custom benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o benchmarks/$@ $(ROW_BREAKDOWN_CUSTOM_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Custom Row-Parallel Breakdown build complete."
	@echo "\nRun: mpirun -np 2 ./benchmarks/row_parallel_breakdown_custom\n"

# ==================== EXAMPLES ====================
EXAMPLE_OBJDIR := $(LIB_OBJDIR)/examples

$(EXAMPLE_OBJDIR)/%.o: examples/%.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

gradient_sync_example: $(EXAMPLE_OBJDIR)/gradient_sync_example.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o examples/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Gradient Sync Example build complete."
	@echo "\nRun: mpirun -np 4 ./examples/gradient_sync_example\n"

tensor_parallel_mlp: $(EXAMPLE_OBJDIR)/tensor_parallel_mlp.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o examples/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Tensor Parallel MLP Example build complete."
	@echo "\nRun: mpirun -np 2 ./examples/tensor_parallel_mlp\n"

# ==================== MORE TESTS ====================
.PHONY: test_mlp_benchmark test_dtensor_factories test_dtensor_creation_methods test_rotate3d_sharding

test_mlp_benchmark: $(TEST_OBJDIR)/test_mlp_benchmark.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] MLP Benchmark build complete."
	@echo "\nRun: mpirun -np 2 ./tests/test_mlp_benchmark\n"

test_dtensor_factories: $(TEST_OBJDIR)/test_dtensor_factories.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] DTensor Factory test build complete."
	@echo "\nRun: mpirun -np 2 ./tests/test_dtensor_factories\n"

test_dtensor_creation_methods: $(TEST_OBJDIR)/test_dtensor_creation_methods.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] DTensor Creation Methods test build complete."
	@echo "\nRun: mpirun -np 2 ./tests/test_dtensor_creation_methods\n"

test_rotate3d_sharding: $(TEST_OBJDIR)/test_rotate3d_sharding.o $(LIB_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o tests/$@ $< $(LIB_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Rotate3D Sharding test build complete."
	@echo "\nRun: mpirun -np 2 ./tests/test_rotate3d_sharding\n"