CXX = /usr/bin/mpic++
NVCC = /usr/local/cuda/bin/nvcc
AR = ar

CXXFLAGS += -fPIC -std=c++17 -O3 -g -DWITH_CUDA
NVCC_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -arch=sm_86 -ccbin=$(CXX)

CUDA_HOME := /usr/local/cuda

TENSOR_IMPL_DIR := $(CURDIR)/../../Tensor-Implementations

INCLUDES = \
    -I. \
    -I./tensor \
    -I./dnn \
    -I$(TENSOR_IMPL_DIR)/include \
    -I$(CUDA_HOME)/include

# Object output directory
OBJDIR := ../lib/tensor-parallelism

# Core library sources
LIB_CPP_SRCS = \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    tensor/placement.cpp

LIB_CUDA_SRCS = \
    dnn/dist_grad_norm_kernels.cu \
    dnn/EntropyKernels.cu \
    tensor/headtail_kernel.cu \
    tensor/fused_transpose_kernel.cu

LIB_CPP_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(LIB_CPP_SRCS))
LIB_CUDA_OBJS = $(patsubst %.cu,$(OBJDIR)/%.o,$(LIB_CUDA_SRCS))
LIB_OBJS = $(LIB_CPP_OBJS) $(LIB_CUDA_OBJS)

LIB_A  = $(OBJDIR)/libdtensor.a
LIB_SO = $(OBJDIR)/libdtensor.so

SHARED_LDFLAGS = -shared -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcurand -lnccl -lmpi -lgomp -lpthread -ldl -lz

.PHONY: all lib clean

all: lib

lib: $(LIB_A) $(LIB_SO)

$(LIB_A): $(LIB_OBJS)
	@echo "\n[AR] Creating static library: $@"
	$(AR) rcs $@ $(LIB_OBJS)
	@echo "[SUCCESS] $(notdir $(LIB_A)) built successfully."

$(LIB_SO): $(LIB_OBJS)
	@echo "\n[LINK] Creating shared library: $@"
	$(CXX) $(CXXFLAGS) -shared -o $@ $(LIB_OBJS) $(SHARED_LDFLAGS)
	@echo "[SUCCESS] $(notdir $(LIB_SO)) built successfully."

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "[COMPILE CUDA] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	@echo "[CLEAN] Removing object files and libraries..."
	rm -rf $(OBJDIR)
