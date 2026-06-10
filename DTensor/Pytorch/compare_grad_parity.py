"""
Gradient-parity check for Q1: which CP backward is mathematically exact?

Reference = single-GPU (cp=1) first-layer c_attn.weight.grad (exact full attention).
For each 2-GPU config, AVG(rank0, rank1) grad should EQUAL the reference iff the
CP backward is correct (params replicated; each rank sums over its 512-token shard;
AVG over 2 ranks == single-GPU mean over 1024 tokens).

Predicted outcomes:
  - legacy : missing cross-rank dK/dV term  -> some deviation from ref
  - attnstyle: if correct -> ~0 deviation; if Path-A backward is biased -> LARGER
               deviation than legacy (which would explain worse convergence).

Run the 3 dumps first (GRAD_PARITY_DUMP=1), then: python3 compare_grad_parity.py
"""
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    p = os.path.join(HERE, name)
    return np.load(p) if os.path.exists(p) else None

def rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

def cosine(a, b):
    af, bf = a.ravel(), b.ravel()
    return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))

def best_scale(combined, ref):
    # scalar s minimizing ||s*combined - ref||
    c = combined.ravel(); r = ref.ravel()
    s = float(c @ r / (c @ c + 1e-30))
    resid = np.linalg.norm(s * c - r) / (np.linalg.norm(r) + 1e-30)
    return s, float(resid)

ref = load("gradparity_legacy_cp1_rank0.npy")
if ref is None:
    ref = load("gradparity_attnstyle_cp1_rank0.npy")
if ref is None:
    print("ERROR: no single-GPU reference (gradparity_*_cp1_rank0.npy). "
          "Run a cp=1 dump first."); raise SystemExit(1)

print(f"reference (cp=1 single-GPU exact)  L2={np.linalg.norm(ref):.6e}  shape={ref.shape}\n")

for tag in ["legacy", "attnstyle"]:
    g0 = load(f"gradparity_{tag}_cp2_rank0.npy")
    g1 = load(f"gradparity_{tag}_cp2_rank1.npy")
    if g0 is None or g1 is None:
        print(f"[{tag}] missing cp2 rank dumps, skipping"); continue
    avg = 0.5 * (g0 + g1)
    s, resid = best_scale(avg, ref)
    print(f"[{tag} cp=2]  AVG(r0,r1) vs single-GPU ref:")
    print(f"    rel_err(AVG, ref)        = {rel_err(avg, ref):.4%}")
    print(f"    cosine(AVG, ref)         = {cosine(avg, ref):.6f}")
    print(f"    best scalar s            = {s:.4f}  (1.0 = perfect scale)")
    print(f"    rel_err after best-scale = {resid:.4%}  (structural error, scale removed)")
    print(f"    L2(r0)={np.linalg.norm(g0):.4e}  L2(r1)={np.linalg.norm(g1):.4e}  L2(avg)={np.linalg.norm(avg):.4e}\n")

print("VERDICT GUIDE:")
print("  - Whichever config has SMALLER rel_err/structural-err is the more-correct backward.")
print("  - If attnstyle > legacy error -> attnstyle's Path-A backward is biased (explains worse loss).")
print("  - If best scalar s != 1 but structural err ~0 -> pure normalization/scale issue, not bias.")
