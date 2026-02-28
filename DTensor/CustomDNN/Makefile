# =============================================================================
# CustomDNN - Backend Makefile
# Builds: customdnn.a (static library)
# Builds the DTensor core + CustomDNN module into one library
# =============================================================================

CXX = /home/blu-bridge25/.local/bin/mpic++

NVCC_LINKER = /usr/local/cuda/bin/nvcc
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -G -arch=sm_86 -ccbin=$(CXX)

CXXFLAGS += -fPIC -std=c++17 -O3 -g -DWITH_CUDA

NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -G -arch=sm_86 -ccbin=/home/blu-bridge25/.local/bin/mpic++

TARGET = main

CUDA_HOME      := /usr/local/cuda
TENSOR_LIB_DIR := /home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Tensor-Implementations
TENSOR_LIB_A   := /home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Tensor-Implementations/lib/libtensor.a


# --- Include Directories ---
INCLUDES = \
    -I. \
    -I$(DTENSOR_DIR) \
    -I$(DTENSOR_DIR)/tensor \
    -I$(DTENSOR_DIR)/process_group \
    -I$(DTENSOR_DIR)/dnn \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include

# --- Library Directories ---
LIB_PATHS = \
    -L$(TENSOR_LIB_DIR)/lib \
    -L$(CUDA_HOME)/lib64 \
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib

# --- Libraries ---
LIBS = \
    -lmpi \
    -lnccl \
    -lcudart \
    -lcublas \
    -lcurand \
    -lnvToolsExt \
    -lgomp \
    -lstdc++ \
    -lz -lpthread -ldl


LDFLAGS = -L/home/blu-bridge25/.local/lib -Xlinker -rpath -Xlinker /home/blu-bridge25/.local/lib

# =============================================================================
# LIBRARY
# =============================================================================
LIBDIR := lib
LIB_A  := $(LIBDIR)/customdnn.a

# DTensor core sources (compiled from DTensor directory)
DTENSOR_CPP_SRCS := \
    $(DTENSOR_DIR)/tensor/dtensor.cpp \
    $(DTENSOR_DIR)/tensor/device_mesh.cpp \
    $(DTENSOR_DIR)/tensor/placement.cpp \
    $(DTENSOR_DIR)/process_group/processGroupNCCL.cpp

DTENSOR_CU_SRCS := \
    $(DTENSOR_DIR)/process_group/fused_transpose_kernel.cu \
    $(DTENSOR_DIR)/dnn/dist_grad_norm_kernels.cu \
    $(DTENSOR_DIR)/TrainingScripts/EntropyKernels.cu

# CustomDNN sources
CUSTOM_SRCS := CustomDNN.cpp

# Object files go into lib/objects/
LIB_OBJDIR := $(LIBDIR)/objects

# DTensor objects (preserve directory structure)
DTENSOR_CPP_OBJS := $(patsubst $(DTENSOR_DIR)/%.cpp,$(LIB_OBJDIR)/dtensor/%.o,$(DTENSOR_CPP_SRCS))
DTENSOR_CU_OBJS  := $(patsubst $(DTENSOR_DIR)/%.cu,$(LIB_OBJDIR)/dtensor/%.o,$(DTENSOR_CU_SRCS))

# CustomDNN objects
CUSTOM_OBJS := $(patsubst %.cpp,$(LIB_OBJDIR)/%.o,$(CUSTOM_SRCS))

ALL_OBJS := $(DTENSOR_CPP_OBJS) $(DTENSOR_CU_OBJS) $(CUSTOM_OBJS)

# =============================================================================
# BUILD TARGETS
# =============================================================================

.PHONY: all lib clean help

all: lib
	@echo "\n[SUCCESS] CustomDNN library built: $(LIB_A)"
	@echo "For tests: cd gpt2_tests && make gpt2_test (or make gpt2_tp_test)"

lib: $(LIB_A)

help:
	@echo "CustomDNN Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build customdnn.a (includes DTensor core)"
	@echo "  make clean    - Remove all build artifacts"
	@echo ""
	@echo "For tests: cd gpt2_tests && make"

gpt2_test:
	cd gpt2_tests && $(MAKE) gpt2_test

gpt2_tp_test:
	cd gpt2_tests && $(MAKE) gpt2_tp_test

# =============================================================================
# COMPILATION RULES
# =============================================================================

# DTensor C++ sources
$(LIB_OBJDIR)/dtensor/%.o: $(DTENSOR_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# DTensor CUDA sources
$(LIB_OBJDIR)/dtensor/%.o: $(DTENSOR_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "[NVCC COMPILE] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# CustomDNN C++ sources
$(LIB_OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# =============================================================================
# LIBRARY BUILD
# =============================================================================

$(LIB_A): $(ALL_OBJS) $(TENSOR_LIB_A)
	@mkdir -p $(LIBDIR)
	@echo "\n[ARCHIVE] Creating static library: $@"
	ar rcs $@ $(ALL_OBJS)
	@echo "[SUCCESS] Static library created: $@"

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "\n[ERROR] TensorLib not found: $(TENSOR_LIB_A)"; \
		echo "        Please build Tensor-Implementations first.\n"; \
		exit 1; \
	fi

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -rf $(LIBDIR)
	@echo "[CLEAN] Done"
