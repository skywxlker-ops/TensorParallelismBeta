#!/usr/bin/env bash
set -euo pipefail

# ---------- User knobs (edit or pass via env/CLI) ----------
NP="${NP:-2}"                         # number of ranks / GPUs
SCRIPT="${SCRIPT:-examples/tp_mlp_fwbw_matmul.py}"

BATCH="${BATCH:-8}"
HIDDEN="${HIDDEN:-1024}"
FFN="${FFN:-4096}"
STEPS="${STEPS:-3}"

LR="${LR:-1e-3}"
WD="${WD:-1e-2}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
EPS="${EPS:-1e-8}"

# Optional: choose specific GPUs. For 2 ranks, e.g. "0,1"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ---------- Environment ----------
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$REPO_DIR/python:$PYTHONPATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# NCCL & MPI quality-of-life (safe defaults for single node)
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"   # good on NVLink boxes
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ---------- Logging ----------
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-$REPO_DIR/logs}"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/tp_${TS}.log"

echo "[run_tp] NP=$NP  GPUs=$CUDA_VISIBLE_DEVICES  script=$SCRIPT"
echo "[run_tp] logging to $LOGFILE"

# ---------- Run ----------
mpirun -np "$NP" \
  -x PYTHONPATH -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
  -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_LEVEL -x OMP_NUM_THREADS \
  /usr/bin/python3 "$SCRIPT" \
    --batch "$BATCH" \
    --hidden "$HIDDEN" \
    --ffn "$FFN" \
    --steps "$STEPS" \
    --lr "$LR" \
    --wd "$WD" \
    --beta1 "$BETA1" \
    --beta2 "$BETA2" \
    --eps "$EPS" \
  2>&1 | tee "$LOGFILE"
