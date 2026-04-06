#!/bin/bash
# Simple script to test ReduceScatter and Broadcast with smaller sizes

cd "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor"

echo "Running full benchmark (this will take ~2-3 minutes)..."
echo "Note: Large AllGather tests may OOM - this is expected"
echo ""

mpirun -np 2 ./benchmarks/nccl_benchmark 2>&1 | grep -E "(===|---| MB |\| Skipped)"
