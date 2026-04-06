#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — Evaluation harness for autoresearch
#
# Builds and runs the CP ring attention benchmark.
# Outputs a SINGLE number to stdout: the METRIC_MS value.
#
# Exit codes:
#   0  → success, metric printed
#   1  → build failure
#   2  → runtime failure
#
# Usage:
#   ./benchmark.sh           # build + run
#   ./benchmark.sh --no-build   # skip build (use existing binary)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD=true
if [[ "${1:-}" == "--no-build" ]]; then
    BUILD=false
fi

# --- Build ---
if $BUILD; then
    echo "[benchmark] Building cp_ring_attn_bench..." >&2
    if ! make -j$(nproc) cp_ring_attn_bench >/dev/null 2>&1; then
        echo "[benchmark] BUILD FAILED" >&2
        echo "99999.0"
        exit 1
    fi
    echo "[benchmark] Build OK" >&2
fi

# --- Run ---
echo "[benchmark] Running: mpirun -np 2 ./cp_ring_attn_bench_exec" >&2

OUTPUT=$(mpirun -np 2 ./cp_ring_attn_bench_exec 2>/dev/null) || {
    echo "[benchmark] RUNTIME FAILURE" >&2
    echo "99999.0"
    exit 2
}

# --- Extract the single metric ---
METRIC=$(echo "$OUTPUT" | grep '^METRIC_MS=' | head -1 | cut -d= -f2)

if [[ -z "$METRIC" ]]; then
    echo "[benchmark] Could not parse METRIC_MS from output" >&2
    echo "[benchmark] Raw output:" >&2
    echo "$OUTPUT" >&2
    echo "99999.0"
    exit 2
fi

# Print detail to stderr for human inspection
echo "$OUTPUT" | grep -E '^(CORRECT|FWD_MS|BWD_MS|METRIC_MS)=' >&2

# THE SINGLE NUMBER — this is what autoresearch reads from stdout
echo "$METRIC"
