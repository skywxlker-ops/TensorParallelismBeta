#!/usr/bin/env python3
"""
Compare step-0 gradient MAGNITUDES between PyTorch (attnstyle) and C++.

Matches strictly BY NAME (not by position) because the two implementations
register parameters in different orders. For every PyTorch param it requires a
C++ entry with the same name AND the same numel; any missing name or numel
mismatch is a hard error -- that is how you know the right tensors are lined up.

L2 magnitude is invariant to the [out,in] vs [in,out] weight-transpose, so the
per-layer ratio is meaningful without any layout handling.
"""
import pickle
import sys
import numpy as np

PT_PKL = "step0_grads_pt_attnstyle.pkl"
CPP_TXT = "step0_grads_cpp.txt"

# ---- load PyTorch: name -> (numel, L2) ----
with open(PT_PKL, "rb") as f:
    pt_grads = pickle.load(f)
pt = {}
for name, g in pt_grads.items():
    flat = g.reshape(-1).astype(np.float64)
    pt[name] = (flat.size, float(np.linalg.norm(flat)))

# ---- load C++: name -> (numel, L2) ----
cpp = {}
cpp_total = None
with open(CPP_TXT) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split()
            if len(parts) == 3 and parts[1] == "total_L2":
                cpp_total = float(parts[2])
            continue
        name, numel, l2 = line.split()
        cpp[name] = (int(numel), float(l2))

# ---- alignment check: every PT name must exist in C++ with same numel ----
errors = []
pt_only = sorted(set(pt) - set(cpp))
cpp_only = sorted(set(cpp) - set(pt))
if pt_only:
    errors.append(f"Names in PT but NOT C++: {pt_only}")
if cpp_only:
    errors.append(f"Names in C++ but NOT PT: {cpp_only}")
for name in sorted(set(pt) & set(cpp)):
    if pt[name][0] != cpp[name][0]:
        errors.append(f"numel mismatch '{name}': PT={pt[name][0]} C++={cpp[name][0]}")

if errors:
    print("!!! ALIGNMENT FAILED -- you are NOT comparing the right tensors:\n")
    for e in errors:
        print("  " + e)
    print("\nFix the C++ dump names / model config before trusting any ratio.")
    sys.exit(1)

print("ALIGNMENT OK: all", len(pt), "params matched by name with identical numel.\n")

# ---- per-layer magnitude ratio ----
print(f"{'Layer':<42} {'PT L2':<14} {'C++ L2':<14} {'Ratio C++/PT':<12}")
print("-" * 84)
tot_pt2 = tot_cpp2 = 0.0
for name in pt:  # PT iteration order, just for display
    n, pt_l2 = pt[name]
    _, cpp_l2 = cpp[name]
    ratio = cpp_l2 / pt_l2 if pt_l2 > 0 else float("nan")
    tot_pt2 += pt_l2 ** 2
    tot_cpp2 += cpp_l2 ** 2
    flag = "  <== " + ("HIGH" if ratio > 1.5 else "") if ratio > 1.5 else ""
    print(f"{name:<42} {pt_l2:<14.6e} {cpp_l2:<14.6e} {ratio:<12.4f}{flag}")

tot_pt = np.sqrt(tot_pt2)
tot_cpp = np.sqrt(tot_cpp2)
print("-" * 84)
print(f"{'TOTAL':<42} {tot_pt:<14.6e} {tot_cpp:<14.6e} {tot_cpp/tot_pt:<12.4f}")
if cpp_total is not None:
    print(f"(C++ self-reported total_L2 = {cpp_total:.6e}; "
          f"sanity vs recomputed {tot_cpp:.6e})")

print("\nReading the result:")
print("  ratio ~1.0          -> gradient magnitude matches PT (norm rise is benign)")
print("  dQ-path params ~1.0 but dK/dV-influenced (c_attn/c_proj) ~2.0 -> dKV backward over-scales")
print("  uniform ~2.0        -> a global 1/world_size-type factor is missing")

# ---- raw per-element cosine check on two representative tensors ----
import os as _os

def _cosine_check(label, pt_npy, cpp_bin, shape):
    if not (_os.path.exists(pt_npy) and _os.path.exists(cpp_bin)):
        print(f"  [{label}] raw files missing ({pt_npy} / {cpp_bin}) -- skip")
        return
    a = np.load(pt_npy).reshape(-1).astype(np.float64)
    b = np.fromfile(cpp_bin, dtype=np.float32).astype(np.float64)
    if a.size != b.size:
        print(f"  [{label}] SIZE MISMATCH pt={a.size} cpp={b.size} "
              f"(expected {np.prod(shape)}) -- transpose/shape wrong")
        return
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
    print(f"  [{label}] cosine={cos:.6f}  |PT|={na:.6e}  |C++|={nb:.6e}  "
          f"ratio={nb/na:.6f}")

print("\n" + "=" * 84)
print("RAW PER-ELEMENT COSINE CHECK (direction vs magnitude)")
print("=" * 84)
_cosine_check("ln_f.weight (ratio~1.2)", "raw_ln_f_weight_pt.npy",
              "step0_raw_lnf_cpp.bin", (768,))
_cosine_check("h.0.c_attn.weight (ratio~0.004)", "raw_h0_c_attn_weight_pt.npy",
              "step0_raw_c_attn_cpp.bin", (2304, 768))
print("\n  cosine ~1.0 on both -> pure MAGNITUDE difference (Adam largely absorbs)")
print("  cosine < 1.0 on c_attn.weight -> real DIRECTIONAL bug in dW backward")

# ============================================================================
# FULL per-element cosine over ALL params (backward parity).
# Reads step0_grads_cpp_raw.bin (written in CPP_TXT line order, PT [out,in]
# layout — Linear weights already transposed C++-side) and cosines each against
# the PT pickle's full arrays.
# ============================================================================
import os
CPP_RAW = "step0_grads_cpp_raw.bin"
if os.path.exists(CPP_RAW):
    print("\n" + "=" * 84)
    print("FULL BACKWARD COSINE: per-element cosine for ALL params (cos~1.0 => match)")
    print("=" * 84)
    # ordered (name, numel) from the txt
    order = []
    with open(CPP_TXT) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nm, ne, _l2 = line.split()
            order.append((nm, int(ne)))
    blob = np.fromfile(CPP_RAW, dtype=np.float32).astype(np.float64)
    off = 0
    rows = []
    for nm, ne in order:
        cvec = blob[off:off + ne]; off += ne
        pvec = pt_grads[nm].reshape(-1).astype(np.float64)
        if pvec.size != ne:
            rows.append((nm, ne, float("nan"), "SIZE MISMATCH")); continue
        denom = np.linalg.norm(pvec) * np.linalg.norm(cvec)
        cos = float(pvec @ cvec / denom) if denom > 0 else 1.0
        rows.append((nm, ne, cos, ""))
    if off != blob.size:
        print(f"  WARN: blob had {blob.size} floats, consumed {off}")
    rows_sorted = sorted(rows, key=lambda r: (r[2] if r[2] == r[2] else -1))
    print(f"  params compared: {len(rows)}")
    cosvals = np.array([r[2] for r in rows if r[2] == r[2]])
    print(f"  cosine  min={cosvals.min():.6f}  mean={cosvals.mean():.6f}  "
          f"#(<0.999)={int((cosvals < 0.999).sum())}")
    print("  10 LOWEST-cosine params:")
    for nm, ne, cos, note in rows_sorted[:10]:
        print(f"    {cos:+.6f}  {nm}  (numel={ne}) {note}")
else:
    print("\n(no step0_grads_cpp_raw.bin -> rebuild C++ for full backward cosine)")
