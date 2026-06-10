"""
Cosine-only check of the all-reduce question (no training needed).

all-reduce only swaps each rank's final accumulated grad for AVG(g0,g1) AFTER
the micro-loop, so g0/g1 are unchanged. Thus from the existing dumps:
  all-reduce OFF -> each replica steps its own g0 / g1
  all-reduce ON  -> both replicas step AVG(g0,g1)

We compare gradient DIRECTION to the single-GPU exact reference, per rank and
for the AVG, for legacy and attnstyle. cosine(g0,g1) measures how divergent the
two replicas' steps are (the "ensemble" driver).
"""
import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
def load(n):
    p = os.path.join(HERE, n); return np.load(p).ravel() if os.path.exists(p) else None
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

ref = load("gradparity_legacy_cp1_rank0.npy")
print(f"reference = single-GPU exact (cp=1)   L2={np.linalg.norm(ref):.4e}\n")

for tag in ["legacy", "attnstyle"]:
    g0 = load(f"gradparity_{tag}_cp2_rank0.npy")
    g1 = load(f"gradparity_{tag}_cp2_rank1.npy")
    if g0 is None or g1 is None:
        print(f"[{tag}] missing dumps"); continue
    avg = 0.5 * (g0 + g1)
    print(f"=== {tag} ===")
    print(f"  cos(g0,  ref) = {cos(g0, ref):.6f}   [all-reduce OFF: rank0 replica step]")
    print(f"  cos(g1,  ref) = {cos(g1, ref):.6f}   [all-reduce OFF: rank1 replica step]")
    print(f"  cos(AVG, ref) = {cos(avg, ref):.6f}   [all-reduce ON : both replicas step this]")
    print(f"  cos(g0,  g1)  = {cos(g0, g1):.6f}   [replica divergence: 1.0=identical steps]")
    print(f"  |g0|={np.linalg.norm(g0):.4e}  |g1|={np.linalg.norm(g1):.4e}  |AVG|={np.linalg.norm(avg):.4e}\n")

print("READ:")
print("  - If cos(AVG,ref) > cos(g0,ref),cos(g1,ref): all-reduce gives a STRAIGHTER")
print("    step -> should train >= OFF on direction grounds.")
print("  - If cos(g0,g1) is well below 1.0: the two OFF replicas take genuinely")
print("    different directions -> that's the ensemble/divergence effect, which")
print("    cosine cannot score for loss (needs the run). But it quantifies how")
print("    much room there is for an ensemble benefit.")
