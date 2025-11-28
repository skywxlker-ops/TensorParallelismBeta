# NCCL Bootstrap Examples

This directory contains standalone examples demonstrating different NCCL bootstrap transport mechanisms.

## Overview

NCCL bootstrap is the initial setup phase where distributed processes discover each other and establish communication channels. These examples show different ways to accomplish this.

## Examples

### 1. **MPI Bootstrap** (`example_mpi_bootstrap.cpp`)

- Uses MPI for out-of-band communication
- Most common in HPC environments
- Simple and robust

**Compile:**

```bash
mpic++ -o mpi_bootstrap example_mpi_bootstrap.cpp -lnccl -lcudart
```

**Run:**

```bash
mpirun -np 4 ./mpi_bootstrap
```

---

### 2. **File-Based Bootstrap** (`example_file_bootstrap.cpp`)

- Uses shared filesystem for coordination
- Great for debugging and SLURM environments
- No MPI dependency required

**Compile:**

```bash
nvcc -o file_bootstrap example_file_bootstrap.cpp -lnccl
```

**Run:**

```bash
# Terminal 1 (Rank 0)
./file_bootstrap --rank=0 --world-size=4

# Terminal 2 (Rank 1)
./file_bootstrap --rank=1 --world-size=4

# Terminal 3 (Rank 2)
./file_bootstrap --rank=2 --world-size=4

# Terminal 4 (Rank 3)
./file_bootstrap --rank=3 --world-size=4
```

---

### 3. **Environment Variable Bootstrap** (`example_env_bootstrap.cpp`)

- Uses environment variables for coordination
- Ideal for containerized environments (Docker, Kubernetes)
- Compatible with job schedulers

**Compile:**

```bash
nvcc -o env_bootstrap example_env_bootstrap.cpp -lnccl
```

**Run:**

```bash
# First, generate and export NCCL_COMM_ID from rank 0
python3 generate_nccl_id.py

# Then launch all ranks with the same NCCL_COMM_ID
export NCCL_COMM_ID=<generated_hex_string>
./env_bootstrap --rank=0 --world-size=4 &
./env_bootstrap --rank=1 --world-size=4 &
./env_bootstrap --rank=2 --world-size=4 &
./env_bootstrap --rank=3 --world-size=4 &
```

---

### 4. **TCP Socket Bootstrap** (`example_tcp_bootstrap.cpp`)

- Direct TCP socket communication
- Shows how NCCL's default bootstrap works internally
- Manual rank 0 server implementation

**Compile:**

```bash
nvcc -o tcp_bootstrap example_tcp_bootstrap.cpp -lnccl -lpthread
```

**Run:**

```bash
# Terminal 1 (Rank 0 - Server)
./tcp_bootstrap --rank=0 --world-size=4 --host=192.168.1.10

# Other terminals (Clients)
./tcp_bootstrap --rank=1 --world-size=4 --host=192.168.1.10
./tcp_bootstrap --rank=2 --world-size=4 --host=192.168.1.10
./tcp_bootstrap --rank=3 --world-size=4 --host=192.168.1.10
```

---

### 5. **Single-Node Multi-GPU** (`example_single_node.cpp`)

- Uses NCCL's automatic shared memory optimization
- Simplest case - all GPUs on one machine
- No network configuration needed

**Compile:**

```bash
nvcc -o single_node example_single_node.cpp -lnccl -fopenmp
```

**Run:**

```bash
./single_node --num-gpus=8
```

---

## Comparison

| Method | Complexity | Use Case | Dependencies |
|--------|-----------|----------|--------------|
| MPI | Low | HPC clusters | MPI runtime |
| File-Based | Low | Debugging, SLURM | Shared FS |
| Environment | Medium | Kubernetes, Docker | None |
| TCP | High | Custom setups | None |
| Single-Node | Very Low | Local multi-GPU | None |

## Key Concepts

### ncclUniqueId

- 128-byte opaque structure
- Contains TCP connection information
- Must be shared among all ranks

### Bootstrap Flow

1. **Rank 0** generates `ncclUniqueId`
2. **Out-of-band** communication distributes ID to all ranks
3. **All ranks** call `ncclCommInitRank()` with shared ID
4. **NCCL internally** connects ranks via TCP
5. **NCCL sets up** high-performance GPU communication

### Environment Variables

```bash
# Network interface for bootstrap
export NCCL_SOCKET_IFNAME=eth0

# Bootstrap timeout (ms)
export NCCL_COMM_TIMEOUT=10000

# Debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Force shared memory on/off
export NCCL_SHM_DISABLE=0  # 0=enable, 1=disable
```

## Testing

Each example includes a simple AllReduce test to verify the communicator is working correctly.

Expected output:

```
[Rank 0] Bootstrap: SUCCESS
[Rank 0] AllReduce test: PASSED (got 6.0, expected 6.0)
```

## Troubleshooting

### "Connection refused"

- Ensure all ranks use the same `ncclUniqueId`
- Check firewall settings
- Verify `NCCL_SOCKET_IFNAME` is correct

### "Timeout"

- Increase `NCCL_COMM_TIMEOUT`
- Ensure all ranks are started
- Check network connectivity

### "Invalid usage"

- Verify `world_size` matches across ranks
- Ensure unique rank IDs (0 to world_size-1)
- Check GPU availability

## Further Reading

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
