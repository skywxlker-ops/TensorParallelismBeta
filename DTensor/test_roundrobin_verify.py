"""
test_roundrobin_verify.py

TDD verification suite for the Round-Robin sub-chunking + tiling implementation.

Baseline: CP_Training_log156.csv  (load_balance=true, use_roundrobin=false)
New run:  CP_Training_log_rr.csv  (load_balance=true, use_roundrobin=true)
          -- generated after implementing ContextParallel round-robin changes

How to generate the new run CSV:
    Build gpt2_cp_test with use_roundrobin=true, run training, save CSV as
    DTensor/CP_Training_logs/CP_Training_log_rr.csv

Test groups:
    1. CSV existence / schema  (smoke tests)
    2. Loss curve correctness  (round-robin must be numerically equivalent to baseline)
    3. Throughput improvement  (tok_per_sec must be >= baseline, attn time must drop)

Run with:
    pytest DTensor/test_roundrobin_verify.py -v
"""

import os
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
BASELINE_CSV = os.path.join(_HERE, "CP_Training_logs", "CP_Training_log156.csv")
NEWRUN_CSV   = os.path.join(_HERE, "CP_Training_logs", "CP_Training_log_rr.csv")

# How many early steps to check for loss equivalence.
# Steps 0-49 cover warmup; steps 50-149 cover steady-state.
LOSS_CHECK_STEPS = 150

# Absolute tolerance for loss comparison (fp32 training, different tile size
# can produce tiny numerical differences; 0.01 is generous).
LOSS_ATOL = 0.01

# Fraction of steps over which tok_per_sec must be at least as good as baseline.
# We allow a 2% margin to account for measurement noise.
THROUGHPUT_MARGIN = -0.02   # new >= baseline * (1 + THROUGHPUT_MARGIN)

# attn_cp timer: new run must be faster by at least this fraction on average.
# Conservative: 10% improvement (real expected gain is ~30-50%).
ATTN_SPEEDUP_MIN_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV)
    return df


@pytest.fixture(scope="module")
def newrun() -> pd.DataFrame:
    if not os.path.exists(NEWRUN_CSV):
        pytest.skip(
            f"New-run CSV not found: {NEWRUN_CSV}\n"
            "Build gpt2_cp_test with use_roundrobin=true, run training, "
            "save output as CP_Training_log_rr.csv to un-skip these tests."
        )
    df = pd.read_csv(NEWRUN_CSV)
    return df


# ---------------------------------------------------------------------------
# Group 1: CSV schema smoke tests (RED until file exists)
# ---------------------------------------------------------------------------

class TestSchema:
    """Verify both CSVs have the expected columns and row counts."""

    REQUIRED_COLS = [
        "step", "loss", "tok_per_sec", "timer_attn_cp",
        "timer_fwd", "timer_bwd", "dt_ms",
    ]

    def test_baseline_exists(self):
        assert os.path.exists(BASELINE_CSV), f"Baseline CSV missing: {BASELINE_CSV}"

    def test_baseline_columns(self, baseline):
        for col in self.REQUIRED_COLS:
            assert col in baseline.columns, f"Baseline missing column: {col}"

    def test_baseline_has_enough_steps(self, baseline):
        assert len(baseline) >= LOSS_CHECK_STEPS, (
            f"Baseline has only {len(baseline)} rows, need >= {LOSS_CHECK_STEPS}"
        )

    def test_newrun_exists(self, newrun):
        # This test will be skipped (via fixture) if the file doesn't exist.
        assert os.path.exists(NEWRUN_CSV)

    def test_newrun_columns(self, newrun):
        for col in self.REQUIRED_COLS:
            assert col in newrun.columns, f"New-run CSV missing column: {col}"

    def test_newrun_has_enough_steps(self, newrun):
        assert len(newrun) >= LOSS_CHECK_STEPS, (
            f"New-run CSV has only {len(newrun)} rows, need >= {LOSS_CHECK_STEPS}"
        )


# ---------------------------------------------------------------------------
# Group 2: Loss curve correctness
#
# Round-robin is mathematically equivalent to BOT_HALF masking — only K/V
# slicing changes, not the attention pattern. Loss must match within ATOL.
# ---------------------------------------------------------------------------

class TestLossEquivalence:
    """New-run loss must match baseline within numerical tolerance."""

    def test_step0_loss_matches(self, baseline, newrun):
        bl = float(baseline.loc[0, "loss"])
        nr = float(newrun.loc[0, "loss"])
        assert abs(bl - nr) <= LOSS_ATOL, (
            f"Step 0 loss diverged: baseline={bl:.6f} newrun={nr:.6f} "
            f"diff={abs(bl-nr):.6f} > atol={LOSS_ATOL}"
        )

    def test_early_steps_loss_matches(self, baseline, newrun):
        n = min(LOSS_CHECK_STEPS, len(baseline), len(newrun))
        bl_loss = baseline["loss"].iloc[:n].to_numpy(dtype=float)
        nr_loss = newrun["loss"].iloc[:n].to_numpy(dtype=float)
        max_diff = float(np.max(np.abs(bl_loss - nr_loss)))
        assert max_diff <= LOSS_ATOL, (
            f"Loss diverged within first {n} steps: max_diff={max_diff:.6f} > atol={LOSS_ATOL}"
        )

    def test_loss_monotonically_decreasing_early(self, newrun):
        # Loss should decrease on average over first LOSS_CHECK_STEPS steps
        # (not strictly monotone per step, but the trend must be downward).
        n = min(LOSS_CHECK_STEPS, len(newrun))
        losses = newrun["loss"].iloc[:n].to_numpy(dtype=float)
        first_quarter_mean = losses[:n // 4].mean()
        last_quarter_mean  = losses[3 * n // 4:].mean()
        assert last_quarter_mean < first_quarter_mean, (
            f"Loss not decreasing: early_mean={first_quarter_mean:.4f}, "
            f"late_mean={last_quarter_mean:.4f}"
        )

    def test_no_nan_or_inf_loss(self, newrun):
        n = min(LOSS_CHECK_STEPS, len(newrun))
        losses = newrun["loss"].iloc[:n]
        assert not losses.isna().any(), "NaN found in new-run loss column"
        assert np.isfinite(losses.to_numpy(dtype=float)).all(), "Inf found in new-run loss"


# ---------------------------------------------------------------------------
# Group 3: Throughput improvement
#
# Round-robin reduces FLOPs for source_rank > rank_ steps; we expect:
#   - timer_attn_cp to decrease by >= ATTN_SPEEDUP_MIN_FRACTION
#   - tok_per_sec to be at least as good as baseline (within noise margin)
# ---------------------------------------------------------------------------

class TestThroughput:
    """New-run attention timer must improve; throughput must not regress."""

    # Skip step 0 (slow first-step warm-up) and use steps 1..N-1 for timing.
    TIMING_START = 1

    def _steady_slice(self, df: pd.DataFrame, col: str) -> np.ndarray:
        return df[col].iloc[self.TIMING_START:LOSS_CHECK_STEPS].to_numpy(dtype=float)

    def test_attn_cp_timer_improved(self, baseline, newrun):
        bl_attn = self._steady_slice(baseline, "timer_attn_cp").mean()
        nr_attn = self._steady_slice(newrun,   "timer_attn_cp").mean()
        speedup = (bl_attn - nr_attn) / bl_attn
        assert speedup >= ATTN_SPEEDUP_MIN_FRACTION, (
            f"timer_attn_cp did not improve enough:\n"
            f"  baseline mean = {bl_attn:.2f} ms\n"
            f"  new-run  mean = {nr_attn:.2f} ms\n"
            f"  speedup       = {speedup*100:.1f}% (need >= {ATTN_SPEEDUP_MIN_FRACTION*100:.0f}%)"
        )

    def test_tok_per_sec_not_regressed(self, baseline, newrun):
        bl_tps = self._steady_slice(baseline, "tok_per_sec").mean()
        nr_tps = self._steady_slice(newrun,   "tok_per_sec").mean()
        ratio  = (nr_tps - bl_tps) / bl_tps
        assert ratio >= THROUGHPUT_MARGIN, (
            f"tok_per_sec regressed:\n"
            f"  baseline mean = {bl_tps:.0f}\n"
            f"  new-run  mean = {nr_tps:.0f}\n"
            f"  change        = {ratio*100:.1f}% (allowed >= {THROUGHPUT_MARGIN*100:.0f}%)"
        )

    def test_dt_ms_not_regressed(self, baseline, newrun):
        bl_dt = self._steady_slice(baseline, "dt_ms").mean()
        nr_dt = self._steady_slice(newrun,   "dt_ms").mean()
        ratio = (nr_dt - bl_dt) / bl_dt
        # dt_ms should decrease (lower is better); allow 2% noise
        assert ratio <= 0.02, (
            f"dt_ms increased beyond noise margin:\n"
            f"  baseline mean = {bl_dt:.2f} ms\n"
            f"  new-run  mean = {nr_dt:.2f} ms\n"
            f"  change        = {ratio*100:.1f}% (allowed <= 2%)"
        )


# ---------------------------------------------------------------------------
# Group 4: Attn timer per-step distribution sanity
# ---------------------------------------------------------------------------

class TestTimerDistribution:
    """timer_attn_cp values must be plausible (no outliers from wrong masking)."""

    def test_attn_timer_no_extreme_outliers(self, newrun):
        # No step should have timer_attn_cp > 3x the median (would indicate a
        # compute-path falling back to full T_q x T_k for all steps).
        timers = newrun["timer_attn_cp"].iloc[1:LOSS_CHECK_STEPS].to_numpy(dtype=float)
        median = np.median(timers)
        max_val = timers.max()
        assert max_val <= 3 * median, (
            f"Extreme attn timer outlier: max={max_val:.2f} ms, "
            f"median={median:.2f} ms (max > 3x median)"
        )

    def test_attn_timer_variance_not_exploded(self, newrun):
        timers = newrun["timer_attn_cp"].iloc[1:LOSS_CHECK_STEPS].to_numpy(dtype=float)
        cv = timers.std() / timers.mean()
        assert cv < 0.20, (
            f"timer_attn_cp coefficient of variation too high: {cv:.3f} "
            "(>0.20 suggests unstable sub-chunking)"
        )
