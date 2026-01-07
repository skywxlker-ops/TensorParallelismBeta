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
    process_group/processGroupNccl.cpp \
    process_group/reverse.cu \
    memory/cachingAllocator.cpp \
    bridge/tensor_ops_bridge.cpp

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
# CLEANUP
# =============================================================================

clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -rf $(LIB_OBJDIR) $(LIB_A) $(LIB_SO)
	@echo "[CLEAN] Done"