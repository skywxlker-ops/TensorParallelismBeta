# Compiler settings
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lnccl
INCLUDES = -I/usr/local/cuda/include

# Output directory
BUILD_DIR = build
TARGET = $(BUILD_DIR)/dtensor_exec

# Source files
SRCS = dtensor.cpp launch_dtensor.cpp

# Default rule
all: $(TARGET)

$(TARGET): $(SRCS) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(INCLUDES) $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
