# =============================================================================
# MyBlas Makefile - Simple Build System
# =============================================================================

# Compiler Configuration
CXX = g++
NVCC = nvcc
SRCDIR := src
OBJDIR := build/objects
BUILDDIR := build
LIBNAME := myblas
TARGET_SO := $(BUILDDIR)/lib$(LIBNAME).so

# Compilation Flags
CPPFLAGS = -Iinclude -I/usr/local/cuda/include
CXXFLAGS = -std=c++17 -fPIC -Wall -Wextra -g
NVCCFLAGS = -std=c++17 -Xcompiler="-fPIC" -arch=sm_86 -g

# Linker Flags
RPATH = -Xlinker -rpath -Xlinker '$$ORIGIN' -Xlinker -rpath -Xlinker '$$ORIGIN/build'
LDFLAGS = -L/usr/local/cuda/lib64 -L$(BUILDDIR) $(RPATH)
LDLIBS = -lcudart -lcublas

# =============================================================================
# Source Files - Auto-Discovery
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES := $(shell find $(SRCDIR) -name '*.cu')

OBJECTS_FROM_CPP := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
OBJECTS_FROM_CU := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SOURCES))
ALL_OBJECTS := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU)

# =============================================================================
# Test Programs - Auto-Discovery
# =============================================================================
# Automatically find all .cpp files in examples/ and create test targets
TEST_SOURCES := $(wildcard examples/*.cpp)
TEST_NAMES := $(basename $(notdir $(TEST_SOURCES)))
TEST_EXECUTABLES := $(addprefix $(BUILDDIR)/,$(TEST_NAMES))

# =============================================================================
# Build Targets
# =============================================================================
.PHONY: all lib tests clean rebuild help

all: lib tests

lib: $(TARGET_SO)
	@echo "\n✅ Library built: $(TARGET_SO)"

tests: lib
	@echo "\n--- Building tests from examples/ ---"
	@for test_src in $(TEST_SOURCES); do \
		test_name=$$(basename $$test_src .cpp); \
		echo "Building: $$test_name"; \
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(BUILDDIR)/$$test_name $$test_src $(LDFLAGS) -l$(LIBNAME) $(LDLIBS) 2>&1 | grep -v "warning:" || true; \
	done
	@echo "\n✅ Test building complete (errors ignored for incomplete tests)"

# Build shared library
$(TARGET_SO): $(ALL_OBJECTS)
	@echo "\n--- Linking shared library: $@"
	@mkdir -p $(BUILDDIR)
	$(NVCC) -shared $(NVCCFLAGS) $(ALL_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

# Compile C++ source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX]: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA]: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# Build test executables
$(BUILDDIR)/%: examples/%.cpp $(TARGET_SO)
	@echo "Building test: $@"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -l$(LIBNAME) $(LDLIBS)

# =============================================================================
# Utility Targets
# =============================================================================
rebuild:
	@$(MAKE) clean && $(MAKE) all

clean:
	@echo "--- Cleaning build artifacts ---"
	rm -rf $(OBJDIR) $(BUILDDIR)/*.so $(BUILDDIR)/test_* $(BUILDDIR)/benchmark_*

# Quick test builder - Usage: make test TEST=test_hgemm_v2
test: lib
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make test TEST=test_hgemm_v2"; \
		exit 1; \
	fi
	@echo "--- Building test: $(TEST) ---"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(BUILDDIR)/$(TEST) examples/$(TEST).cpp $(LDFLAGS) -l$(LIBNAME) $(LDLIBS)
	@echo "✅ Built: $(BUILDDIR)/$(TEST)"

# Run a test - Usage: make run TEST=test_hgemm_v2
run: lib
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make run TEST=test_hgemm_v2"; \
		exit 1; \
	fi
	@if [ ! -f $(BUILDDIR)/$(TEST) ]; then \
		echo "Building $(TEST) first..."; \
		$(MAKE) test TEST=$(TEST); \
	fi
	@echo "--- Running $(BUILDDIR)/$(TEST) ---"
	LD_LIBRARY_PATH=$(BUILDDIR):$$LD_LIBRARY_PATH $(BUILDDIR)/$(TEST)

help:
	@echo "MyBlas Makefile - Available targets:"
	@echo ""
	@echo "  make              - Build library and all tests (auto-discovered)"
	@echo "  make lib          - Build only the shared library"
	@echo "  make tests        - Build all test programs from examples/"
	@echo "  make test TEST=<name>  - Build specific test"
	@echo "  make run TEST=<name>   - Build and run specific test"
	@echo "  make rebuild      - Clean and rebuild everything"
	@echo "  make clean        - Remove all build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make test TEST=test_hgemm_v2"
	@echo "  make run TEST=test_hgemm_v2"
	@echo ""
	@echo "Auto-discovered tests: $(TEST_NAMES)"

# Dependency tracking
-include $(ALL_OBJECTS:.o=.d)
