CXX = /home/blu-bridge25/.local/bin/mpic++

NVCC_LINKER = /usr/local/cuda/bin/nvcc
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -G -arch=sm_86 -ccbin=$(CXX)

CXXFLAGS += -fPIC -std=c++17 -O3 -g -DWITH_CUDA $(INCLUDES)

NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -G -arch=sm_86 -ccbin=/home/blu-bridge25/.local/bin/mpic++

TARGET = main

CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := /home/blu-bridge25/Study/Code/avengersassemble/TensorParallelismBeta/DTensor/Tensor-Implementations
TENSOR_LIB_A   := /home/blu-bridge25/Study/Code/avengersassemble/TensorParallelismBeta/DTensor/Tensor-Implementations/lib/libtensor.a

INCLUDES = \
    -I. \
    -I./tensor \
    -I./process_group \
    -I./memory \
    -I./bridge \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include \
#     -I/home/blu-bridge25/Study/Code/TensorParallelismBeta/autograd/cgadimpl/include \
    $(AUTOGRAD_INCLUDES)

LIB_PATHS = \
    -L/home/blu-bridge25/Study/Code/avengersassemble/TensorParallelismBeta/DTensor/Tensor-Implementations/lib \

LIBS = -lmpi -lnccl -lcudart -lcublas -lcurand -lgomp -lstdc++ $(AUTOGRAD_LIBS) -lz -lpthread -ldl

LDFLAGS = -L/home/blu-bridge25/.local/lib -Xlinker -rpath -Xlinker /home/blu-bridge25/.local/lib

SRCS = main.cpp \
	tensor/dtensor.cpp \
	tensor/device_mesh.cpp \
	process_group/processGroupNccl.cpp \
	tensor/placement.cpp

OBJS = $(SRCS:.cpp=.o) $(CUDA_OBJS)

.PHONY: all clean test test_mlp_forward

all: $(TARGET)



$(TARGET): $(OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -L/home/blu-bridge25/.local/lib -Xlinker -rpath -Xlinker /home/blu-bridge25/.local/lib -o $(TARGET) $(OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS) -Xlinker --no-keep-memory -Xlinker --reduce-memory-overheads
	@echo "[SUCCESS] Build complete."



TEST_MLP_OBJS = $(TEST_MLP_SRCS:.cpp=.o)

kkk: $(TEST_MLP_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(TEST_MLP_OBJS) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] Test build complete."




$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] Static library $(TENSOR_LIB_A) not found!"; \
		echo "        Please compile the '/home/blu-bridge25/Study/Code/avengersassemble/TensorParallelismBeta/DTensor/Tensor-Implementations' submodule first.\n"; \
		exit 1; \
	fi


clean:
	@echo "[CLEAN] Removing object files and executables..."
	rm -f $(OBJS) $(TARGET) $(TEST_MLP_OBJS) test_mlp_forward
	rm -f tests/*.o tests/test_device_mesh tests/test_matmul
	rm -f TrainingScripts/*.o
	rm -f benchmarks/*.o benchmarks/nccl_benchmark benchmarks/matmul_benchmark benchmarks/row_parallel_breakdown benchmarks/row_parallel_breakdown_custom benchmarks/shard_benchmarking
	rm -f tensor/dtensor_custom.o bridge/tensor_ops_bridge_custom.o

.PHONY: test_device_mesh nccl_benchmark test_matmul matmul_benchmark row_parallel_breakdown row_parallel_breakdown_custom gradient_sync_example shard_benchmarking

CUDA_SRCS =  process_group/fused_transpose_kernel.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

TP_MLP_SRCS = \
    examples/tensor_parallel_mlp_copy.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
	tensor/placement.cpp 


TP_MLP_OBJS = $(TP_MLP_SRCS:.cpp=.o) $(CUDA_OBJS)

CUDA_LDFLAGS = -L/usr/local/cuda-12.6/include



TrainingScripts/EntropyKernels.o: TrainingScripts/EntropyKernels.cu
	@echo "[COMPILE CUDA] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@


TP_MLP_AUTOGRAD_SRCS = \
    examples/tensor_parallel_mlp.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
	tensor/placement.cpp

TP_MLP_AUTOGRAD_OBJS = $(TP_MLP_AUTOGRAD_SRCS:.cpp=.o) $(CUDA_OBJS)

tensor_parallel_mlp: $(TP_MLP_AUTOGRAD_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(TP_MLP_AUTOGRAD_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Tensor Parallel MLP Example with Autograd build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/tensor_parallel_mlp\n"

# Normal MLP with fixed seed data (no sharding/sync) for comparison
MLP_SEED_SRCS = \
    examples/mlp_seed.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
	tensor/placement.cpp

MLP_SEED_OBJS = $(MLP_SEED_SRCS:.cpp=.o) $(CUDA_OBJS)

mlp_seed: $(MLP_SEED_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(MLP_SEED_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Normal MLP (Fixed Seed) build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/mlp_seed\n"

# Tensor Parallel MLP with fixed seed data (sharding/sync) for comparison
TP_MLP_SEED_SRCS = \
    examples/tensor_parallel_mlp_seed.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
	tensor/placement.cpp

TP_MLP_SEED_OBJS = $(TP_MLP_SEED_SRCS:.cpp=.o) $(CUDA_OBJS)

tensor_parallel_mlp_seed: $(TP_MLP_SEED_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(TP_MLP_SEED_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Tensor Parallel MLP (Fixed Seed) build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/tensor_parallel_mlp_seed\n"

# DModule MLP Example (uses DistributedNN.h)
DMLP_EXAMPLE_SRCS = \
    examples/dmlp_example.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
	tensor/placement.cpp

DMLP_EXAMPLE_OBJS = $(DMLP_EXAMPLE_SRCS:.cpp=.o) $(CUDA_OBJS)

dmlp_example: $(DMLP_EXAMPLE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(DMLP_EXAMPLE_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DModule MLP Example build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/dmlp_example\n"


# # NEW CUDA rule  
$(CUDA_OBJS): %.o: %.cu
	@echo "[COMPILE CUDA] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Shard Benchmarking
SHARD_BENCH_SRCS = \
    benchmarks/shard_benchmarking.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

SHARD_BENCH_OBJS = $(SHARD_BENCH_SRCS:.cpp=.o) $(CUDA_OBJS)

shard_benchmarking: $(SHARD_BENCH_OBJS) $(TENSOR_LIB_A) 
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o benchmarks/$@ $(SHARD_BENCH_OBJS) -Xlinker --start-group  $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Shard Benchmarking build complete."
	@echo "\nRun with: mpirun -np 2 ./benchmarks/shard_benchmarking\n"

# Interconnect Benchmarks
INTERCONNECT_BENCH_SRCS = \
    benchmarks/interconnect_benchmarks.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

INTERCONNECT_BENCH_OBJS = $(INTERCONNECT_BENCH_SRCS:.cpp=.o) $(CUDA_OBJS)

interconnect_benchmarks: $(INTERCONNECT_BENCH_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating benchmark executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o benchmarks/$@ $(INTERCONNECT_BENCH_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Interconnect Benchmarks build complete."
	@echo "\nRun with: mpirun -np 2 ./benchmarks/interconnect_benchmarks\n"


EMBEDDING_EXAMPLE_SRCS = \
    examples/embedding_example.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

EMBEDDING_EXAMPLE_OBJS = $(EMBEDDING_EXAMPLE_SRCS:.cpp=.o) $(CUDA_OBJS)

embedding_example: $(EMBEDDING_EXAMPLE_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(EMBEDDING_EXAMPLE_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] Embedding Example build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/embedding_example\n"

# DMLP Embedding Example
DMLP_EMBEDDING_SRCS = \
    examples/dmlp_embedding_example.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

DMLP_EMBEDDING_OBJS = $(DMLP_EMBEDDING_SRCS:.cpp=.o) $(CUDA_OBJS)

dmlp_embedding_example: $(DMLP_EMBEDDING_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(DMLP_EMBEDDING_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DMLP Embedding Example build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/dmlp_embedding_example\n"

# DMLP Full (Parallel Embedding)
DMLP_FULL_SRCS = \
    examples/dmlp_full.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

DMLP_FULL_OBJS = $(DMLP_FULL_SRCS:.cpp=.o) $(CUDA_OBJS)

dmlp_full: $(DMLP_FULL_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(DMLP_FULL_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DMLP Full build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/dmlp_full\n"

# GPT2 Parallel Test
GPT2_PARALLEL_TEST_SRCS = \
    TrainingScripts/gpt2_parallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_PARALLEL_TEST_OBJS = $(GPT2_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS)

gpt2_parallel_test: $(GPT2_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_parallel_test\n"

# GPT2 Entropy Parallel Test
GPT2_ENTROPY_PARALLEL_TEST_SRCS = \
    TrainingScripts/gpt2_entropy_parallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_ENTROPY_PARALLEL_TEST_OBJS = $(GPT2_ENTROPY_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/EntropyKernels.o

gpt2_entropy_parallel_test: $(GPT2_ENTROPY_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_ENTROPY_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Entropy Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_entropy_parallel_test\n"


GPT2_TP_TEST_SRCS = \
    gpt2_tp_test/gpt2_tp_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_TP_TEST_OBJS = $(GPT2_TP_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/EntropyKernels.o

gpt2_tp_test: $(GPT2_TP_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: gpt2_tp_test_exec"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o gpt2_tp_test_exec $(GPT2_TP_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 TP Test build complete."
	@echo "\nRun with: mpirun -np 2 ./gpt2_tp_test_exec \n"

GPT2_TP_MLP_TEST_SRCS = \
    gpt2_tp_test/gpt2_tp_mlp_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_TP_MLP_TEST_OBJS = $(GPT2_TP_MLP_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/EntropyKernels.o

gpt2_tp_mlp_test: $(GPT2_TP_MLP_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: gpt2_tp_mlp_test_exec"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o gpt2_tp_mlp_test_exec $(GPT2_TP_MLP_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 TP MLP Test build complete."
	@echo "\nRun with: mpirun -np 2 ./gpt2_tp_mlp_test_exec \n"
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@


# GPT2 Sync Verification Test
GPT2_SYNC_VERIFY_TEST_SRCS = \
    TrainingScripts/gpt2_sync_verify_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_SYNC_VERIFY_TEST_OBJS = $(GPT2_SYNC_VERIFY_TEST_SRCS:.cpp=.o) $(CUDA_OBJS) TrainingScripts/MaskOps.o

gpt2_sync_verify_test: $(GPT2_SYNC_VERIFY_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_SYNC_VERIFY_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Sync Verification Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_sync_verify_test\n"

# GPT2 MLP Parallel Test (Non-parallel Embedding/LM Head)
GPT2_MLP_PARALLEL_TEST_SRCS = \
    TrainingScripts/gpt2_mlp_parallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_MLP_PARALLEL_TEST_OBJS = $(GPT2_MLP_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS)

gpt2_mlp_parallel_test: $(GPT2_MLP_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_MLP_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 MLP Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_mlp_parallel_test\n"


# GPT2 Tensor Parallel Test
GPT2_TENSOR_PARALLEL_TEST_SRCS = \
    TrainingScripts/GPT2_TesnorParallel_test.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

GPT2_TENSOR_PARALLEL_TEST_OBJS = $(GPT2_TENSOR_PARALLEL_TEST_SRCS:.cpp=.o) $(CUDA_OBJS)

gpt2_tensor_parallel_test: $(GPT2_TENSOR_PARALLEL_TEST_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating example executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o examples/$@ $(GPT2_TENSOR_PARALLEL_TEST_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] GPT2 Tensor Parallel Test build complete."
	@echo "\nRun with: mpirun -np 2 ./examples/gpt2_tensor_parallel_test\n"

# DTensor Leak Tests
DTENSOR_LEAK_TESTS_SRCS = \
    tests/dtensor_leak_tests/dtensor_benchmark_main.cpp \
    tests/dtensor_leak_tests/test_dtensor_layers.cpp \
    tests/dtensor_leak_tests/test_dtensor_activations.cpp \
    tests/dtensor_leak_tests/test_dtensor_losses.cpp \
    tests/dtensor_leak_tests/test_gpt2_components.cpp \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    process_group/processGroupNccl.cpp \
    tensor/placement.cpp

DTENSOR_LEAK_TESTS_OBJS = $(DTENSOR_LEAK_TESTS_SRCS:.cpp=.o) $(CUDA_OBJS)

tests/dtensor_leak_tests/%.o: tests/dtensor_leak_tests/%.cpp
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I./tests/dtensor_leak_tests -c $< -o $@

dtensor_leak_tests: $(DTENSOR_LEAK_TESTS_OBJS) $(TENSOR_LIB_A)
	@echo "\n[LINKING] Creating test executable: $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) $(LIB_PATHS) -o tests/dtensor_leak_tests/$@ $(DTENSOR_LEAK_TESTS_OBJS) -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group $(LIBS) $(LDFLAGS)
	@echo "[SUCCESS] DTensor Leak Tests build complete."
	@echo "\nRun with: mpirun -np 2 ./tests/dtensor_leak_tests/dtensor_leak_tests\n"

