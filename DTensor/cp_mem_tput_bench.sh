#!/usr/bin/env bash
# =============================================================================
# cp_mem_tput_bench.sh
# -----------------------------------------------------------------------------
# After-each-change gate: runs a 2-step probe of BOTH the C++ (BluTrain) and
# PyTorch CP scripts at ws=2 and prints peak-active memory AND throughput so we
# can confirm a memory change did NOT regress throughput below PyTorch.
#
# Usage:
#   ./cp_mem_tput_bench.sh [LABEL N_EMBD N_LAYER N_HEAD TYING T_CPP]
#   defaults: 124M 768 12 12 1 2048   (ws=2 T=2048 => local seq 1024)
#
# Env passthrough: set CP_RECOMPUTE_K / CP_RECOMPUTE_KV / etc before calling to
# benchmark a specific code path.
# =============================================================================
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="${MEM_SNAPSHOT_DIR:-${DIR}/cp_bench_runs}"
mkdir -p "$SNAP"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

LABEL="${1:-124M}"; E="${2:-768}"; L="${3:-12}"; H="${4:-12}"; TY="${5:-1}"
T="${6:-2048}"   # ws=2 => local seq = T/2

echo "=============================================================="
echo " CP mem+throughput bench: $LABEL (embd=$E layer=$L head=$H tying=$TY) ws=2 T=$T"
echo "   extra C++ env: CP_RECOMPUTE_K=${CP_RECOMPUTE_K:-0} CP_RECOMPUTE_KV=${CP_RECOMPUTE_KV:-unset} CP_NO_SAVE_OUT=${CP_NO_SAVE_OUT:-unset}"
echo "=============================================================="

echo "----- C++ (BluTrain) -----"
CP_MEM_PROBE=1 CP_MODEL_LABEL="$LABEL" CP_N_EMBD="$E" CP_N_LAYER="$L" CP_N_HEAD="$H" \
CP_WEIGHT_TYING="$TY" CP_ROTATOR=alltoall CP_T="$T" MEM_SNAPSHOT_DIR="$SNAP" \
timeout 600 mpirun -np 2 "${DIR}/gpt2_cp_test_exec" 2>&1 \
  | grep -iE "Active \(in use\)|loss:.*tok/sec" \
  | sed -E 's/^/  /'

echo "----- PyTorch -----"
MEM_PROBE=1 MODEL_LABEL="$LABEL" N_EMBD="$E" N_LAYER="$L" N_HEAD="$H" \
WEIGHT_TYING="$TY" ROTATE_METHOD=alltoall T="$T" MEM_SNAPSHOT_DIR="$SNAP" \
timeout 600 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  "${DIR}/Pytorch/gpt2_cp_attnstyle_fp32.py" 2>&1 \
  | grep -iE "MEM_PROBE rank=0|tok/sec" \
  | sed -E 's/^/  /'

echo "=============================================================="
echo " Compare: C++ 'Active peak' vs PyTorch 'peak_alloc_mb' (both live-peak),"
echo "          and C++ tok/sec vs PyTorch tok/sec. Goal: mem down, tput >= PT."
echo "=============================================================="
