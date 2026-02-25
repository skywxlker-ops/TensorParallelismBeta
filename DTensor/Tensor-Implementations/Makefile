# =============================================================================
# Configuration
# =============================================================================
CXX = g++
NVCC = nvcc
SRCDIR := src
OBJDIR := lib/objects
LIBDIR := lib
TARGET_A := $(LIBDIR)/libtensor.a
TARGET_SO := $(LIBDIR)/libtensor.so

# =============================================================================
# CUDA auto-detection (Ubuntu + Arch portable)
# =============================================================================
CUDA_NVCC := $(shell which nvcc 2>/dev/null)

ifeq ($(CUDA_NVCC),)
$(error nvcc not found. CUDA toolkit is not installed or not in PATH)
endif

CUDA_ROOT := $(shell dirname $(shell dirname $(CUDA_NVCC)))
CUDA_INC  := $(CUDA_ROOT)/include
CUDA_LIB  := $(CUDA_ROOT)/lib64

# Auto-detect NVTX include path (CUDA 12+ uses nvtx3 subdirectory)
NVTX_INC := $(shell find $(CUDA_ROOT)/targets -name "nvtx3" -type d 2>/dev/null | head -1)
ifeq ($(NVTX_INC),)
NVTX_INC := $(CUDA_INC)
endif

# Auto-detect NVTX library path (CUDA 12+ uses targets/x86_64-linux/lib)
NVTX_LIB := $(shell find $(CUDA_ROOT)/targets -name "libnvToolsExt.so" -exec dirname {} \; 2>/dev/null | head -1)
ifeq ($(NVTX_LIB),)
NVTX_LIB := $(CUDA_LIB)
endif

CPPFLAGS = -Iinclude -I$(CUDA_INC) -I$(NVTX_INC) -DWITH_CUDA 

# Autograd Profiler flag
ifeq ($(PROFILER),1)
CPPFLAGS += -DAUTOGRAD_PROFILER_ENABLED
endif
CXXFLAGS = -std=c++20 -fPIC -Wall -Wextra -g -O3 -fopenmp
NVCCFLAGS = -std=c++20 -Xcompiler="-fPIC" -arch=sm_86 -O3 --expt-relaxed-constexpr

# Auto-detect CUDA targets lib directory (for NVTX and other libraries)
CUDA_TARGETS_LIB := $(shell find $(CUDA_ROOT)/targets -type d -name "lib" 2>/dev/null | head -1)
ifeq ($(CUDA_TARGETS_LIB),)
CUDA_TARGETS_LIB := $(CUDA_LIB)
endif

RPATH = -Xlinker -rpath -Xlinker '$$ORIGIN/lib'
LDFLAGS = -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(LIBDIR) $(RPATH)
# Note: nvToolsExt library is deprecated in CUDA 12+, NVTX3 is header-only
LDLIBS = -lcudart -ltbb -lcurand -lcublas

# =============================================================================
# File Discovery (Automatic)
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES := $(shell find $(SRCDIR) -name '*.cu')

CU_SOURCES_FOR_DLINK := \
    src/Views/ContiguousKernel.cu \
    src/UnaryOps/cuda/ReductionImplGPU.cu

CU_SOURCES_REGULAR := $(filter-out $(CU_SOURCES_FOR_DLINK), $(CU_SOURCES))
OBJECTS_FROM_CPP := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
OBJECTS_FROM_CU_REGULAR := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_REGULAR))
OBJECTS_FOR_DLINK := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_FOR_DLINK))
DEVICE_LINK_OBJ := $(OBJDIR)/device_link.o
ALL_OBJECTS := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU_REGULAR) $(OBJECTS_FOR_DLINK)

# =============================================================================
# Main Build Rules
# =============================================================================
.PHONY: all
all: $(TARGET_SO) $(TARGET_A)
	@echo "\n\n✅ Library build is up-to-date."

$(TARGET_SO): $(ALL_OBJECTS) $(DEVICE_LINK_OBJ)
	@echo "\n\n--- Creating shared library: $@"
	$(NVCC) -shared $(NVCCFLAGS) $(ALL_OBJECTS) $(DEVICE_LINK_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(TARGET_A): $(ALL_OBJECTS)
	@echo "\n\n--- Creating static library: $@"
	ar rcs $@ $(ALL_OBJECTS)

# --- Compilation Pattern Rules ---
$(DEVICE_LINK_OBJ): $(OBJECTS_FOR_DLINK)
	@echo "\n\n--- Linking CUDA device code..."
	$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX]: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJECTS_FOR_DLINK): NVCC_DEVICE_FLAGS = -dc
$(OBJECTS_FOR_DLINK): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA -dc]: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_DEVICE_FLAGS) -c $< -o $@

$(OBJECTS_FROM_CU_REGULAR): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA]: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# =============================================================================
# Custom Action Rules
# =============================================================================
.PHONY: run-snippet
run-snippet: $(TARGET_SO)
	@if [ -z "$(FILE)" ]; then \
        echo "ERROR: Please specify a file to run."; \
        echo "Usage: make run-snippet FILE=path/to/your/file.cpp [PROFILER=1]"; \
        exit 1; \
    fi
	@echo "--- Compiling snippet: $(FILE) ---"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o snippet_runner $(FILE) $(LDFLAGS) -ltensor $(LDLIBS)
	@echo "\n--- Running snippet_runner ---"
	./snippet_runner
	@echo "\n"
#   @echo "\n--- Cleaning up snippet ---"
#   rm -f snippet_runner

.PHONY: rebuild
rebuild:
	@$(MAKE) clean && $(MAKE) all

.PHONY: clean
clean:
	@echo "--- Cleaning up build files ---"
	rm -rf $(OBJDIR) $(TARGET_A) $(TARGET_SO) 

.PHONY: run-folder
run-folder: $(TARGET_SO)
	@if [ -z "$(FOLDER)" ]; then \
        echo "ERROR: Please specify a folder to run."; \
        echo "Usage: make run-folder FOLDER=Tests/path/to/folder"; \
        exit 1; \
    fi
	@echo "--- Running all tests in: $(FOLDER) ---"
	@for file in $(FOLDER)/*.cpp; do \
        if [ ! -f "$$file" ]; then continue; fi; \
        echo "\n---------------------------------------------------------"; \
        echo "Running: $$file"; \
        echo "---------------------------------------------------------"; \
        $(CXX) $(CPPFLAGS) $(CXXFLAGS) -o snippet_runner "$$file" $(LDFLAGS) -ltensor $(LDLIBS) || exit 1; \
        ./snippet_runner || exit 1; \
    done
	@rm -f snippet_runner
	@echo "\n✅ All tests in $(FOLDER) passed!"