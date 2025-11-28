# Makefile for NCCL Bootstrap Examples

NVCC = nvcc
MPICXX = mpic++
CXXFLAGS = -std=c++11 -O2
NVCCFLAGS = -std=c++11 -O2
LIBS = -lnccl -lcudart
OMP_FLAGS = -Xcompiler -fopenmp

# Targets
ALL_TARGETS = mpi_bootstrap file_bootstrap env_bootstrap single_node

.PHONY: all clean help

all: $(ALL_TARGETS)

# MPI Bootstrap
mpi_bootstrap: example_mpi_bootstrap.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LIBS)

# File-Based Bootstrap
file_bootstrap: example_file_bootstrap.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LIBS)

# Environment Variable Bootstrap
env_bootstrap: example_env_bootstrap.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LIBS)

# Single-Node Multi-GPU
single_node: example_single_node.cpp
	$(NVCC) $(NVCCFLAGS) $(OMP_FLAGS) -o $@ $< $(LIBS)

# Clean
clean:
	rm -f $(ALL_TARGETS)
	rm -f /tmp/nccl_id.bin

# Help
help:
	@echo "NCCL Bootstrap Examples - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all examples"
	@echo "  mpi_bootstrap  - Build MPI bootstrap example"
	@echo "  file_bootstrap - Build file-based bootstrap example"
	@echo "  env_bootstrap  - Build environment variable bootstrap example"
	@echo "  single_node    - Build single-node multi-GPU example"
	@echo "  clean          - Remove all built binaries"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make mpi_bootstrap && mpirun -np 4 ./mpi_bootstrap"
	@echo "  make single_node && ./single_node 8"
	@echo "  make env_bootstrap && ./env_bootstrap --generate"
