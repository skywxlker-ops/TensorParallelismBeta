#!/bin/bash
# Test script to demonstrate dynamic expected value calculation

echo "=========================================="
echo "Testing with 2 GPUs (2 Ranks)"
echo "=========================================="
mpirun -np 2 --allow-run-as-root ./test_mlp_forward 2>&1 | grep -A 6 "Expected Results"

echo ""
echo "=========================================="
echo "Expected formula verification:"
echo "  For 2 ranks: Y2 = 8 * (2 + 1) = 24"
echo "  For 4 ranks: Y2 = 8 * (4 + 1) = 40"
echo "  For 8 ranks: Y2 = 8 * (8 + 1) = 72"
echo "=========================================="
