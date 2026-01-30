# =============================================================================
# Makefile for Bluwerp DTensor Training
# =============================================================================

# --- Parent Directory Variables ---
PARENT_DIR := ..
TENSOR_LIB_DIR := $(PARENT_DIR)/Tensor-Implementations
TENSOR_LIB_A := $(TENSOR_LIB_DIR)/lib/libtensor.a
UNPARALLELED_A := $(PARENT_DIR)/lib/unparalleled.a

# --- Compiler Configuration ---
CXX = mpic++
NVCC_LINKER = /usr/local/cuda/bin/nvcc

# --- Build Flags ---
CXXFLAGS = -std=c++17 -O3 -fPIC -g -DWITH_CUDA
NVCC_LINK_FLAGS = -std=c++17 -Xcompiler -fPIC -ccbin=mpic++

# --- Paths ---
CUDA_HOME := /usr/local/cuda

# --- Include Directories ---
INCLUDES = \
    -I$(PARENT_DIR) \
    -I$(PARENT_DIR)/include \
    -I$(PARENT_DIR)/tensor \
    -I$(PARENT_DIR)/process_group \
    -I$(PARENT_DIR)/memory \
    -I$(PARENT_DIR)/bridge \
    -I$(PARENT_DIR)/ckpt \
    -I$(PARENT_DIR)/nn \
    -I$(TENSOR_LIB_DIR)/include \
    -I$(CUDA_HOME)/include

# --- Library Directories ---
LIB_PATHS = \
    -L$(PARENT_DIR)/lib \
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

# --- Targets ---
TARGET = gpt_train_bluwerp

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean help check_lib

# Default target
all: check_lib $(TARGET)
	@echo "\n[SUCCESS] $(TARGET) built!"

# Check library exists
check_lib:
	@if [ ! -f $(UNPARALLELED_A) ]; then \
		echo "[BUILD] Building unparalleled.a first..."; \
		$(MAKE) -C $(PARENT_DIR) lib; \
	fi
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo "[BUILD] Building libtensor.a first..."; \
		$(MAKE) -C $(TENSOR_LIB_DIR) lib; \
	fi

# Help target
help:
	@echo "Bluwerp DTensor Training Makefile"
	@echo ""
	@echo "Build Targets:"
	@echo "  all                    - Build gpt_train_bluwerp"
	@echo "  clean                  - Remove all built files"
	@echo ""
	@echo "Run with: mpirun -np 2 ./gpt_train_bluwerp"

# Clean target
clean:
	@echo "[CLEAN] Removing executables and objects..."
	rm -f $(TARGET) *.o
	@echo "[CLEAN] Done"

# =============================================================================
# Compilation Rules
# =============================================================================

# Compile .cpp to .o
%.o: %.cpp
	@echo "\n[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# =============================================================================
# Linking Rules
# =============================================================================

$(TARGET): $(TARGET).o $(UNPARALLELED_A) $(TENSOR_LIB_A)
	@echo "\n[LINKING] $@"
	$(NVCC_LINKER) $(NVCC_LINK_FLAGS) -o $@ $(TARGET).o $(UNPARALLELED_A) $(TENSOR_LIB_A) $(LIB_PATHS) $(LIBS)
	@echo "[SUCCESS] $@ built"
	@echo "Run: mpirun -np 2 ./$@\n"
