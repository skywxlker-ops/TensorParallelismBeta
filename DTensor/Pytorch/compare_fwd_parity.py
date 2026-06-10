#!/usr/bin/env python3
"""
Forward-parity check (ws=1 base-model): compares PT vs C++ activations at three
checkpoints to bisect where the forwards first diverge.

  emb  = tok_emb + pos_emb       (input to transformer stack)
  blk0 = output of block 0       (attention + MLP)
  lnf  = output of final LayerNorm (input to lm_head)

All are [B, T, C] = [4, 1024, 768], row-major, fp32 (PT .npy / C++ .bin).
Reports overall cosine + per-token-position cosine summary so a token-ORDER
mismatch (LB reorder) is distinguishable from a value mismatch.
"""
import os
import numpy as np

CKPTS = [
    ("emb  (transformer input)", "fwd_emb_pt.npy", "fwd_emb_cpp.bin"),
    ("sdpa (raw attn out,pre-c_proj)", "fwd_sdpa_pt.npy", "fwd_sdpa_cpp.bin"),
    ("blk0attn (post-attn,pre-MLP)", "fwd_blk0attn_pt.npy", "fwd_blk0attn_cpp.bin"),
    ("blk0 (after block 0)",     "fwd_blk0_pt.npy", "fwd_blk0_cpp.bin"),
    ("lnf  (final hidden)",      "fwd_lnf_pt.npy", "fwd_lnf_cpp.bin"),
]
SHAPE = (4, 1024, 768)

print("=" * 78)
print("FORWARD PARITY (ws=1):  cosine ~1.0 => forwards agree")
print("=" * 78)
for label, pt_f, cpp_f in CKPTS:
    if not (os.path.exists(pt_f) and os.path.exists(cpp_f)):
        print(f"  {label:<26} MISSING ({pt_f} / {cpp_f})")
        continue
    a = np.load(pt_f).astype(np.float64).reshape(-1)
    b = np.fromfile(cpp_f, dtype=np.float32).astype(np.float64)
    if a.size != b.size:
        print(f"  {label:<26} SIZE MISMATCH pt={a.size} cpp={b.size}")
        continue
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    ratio = np.linalg.norm(b) / np.linalg.norm(a)
    # per-token-position cosine (does a token-ordering mismatch hide here?)
    A = a.reshape(SHAPE)[0]  # [T, C], batch 0
    B = b.reshape(SHAPE)[0]
    tok_cos = np.array([
        float(A[t] @ B[t] / (np.linalg.norm(A[t]) * np.linalg.norm(B[t]) + 1e-30))
        for t in range(0, SHAPE[1], 128)
    ])
    print(f"  {label:<26} cosine={cos:+.6f}  ratio={ratio:.4f}")
    print(f"      per-token cos (every 128th): "
          f"{np.array2string(tok_cos, precision=3, max_line_width=200)}")
print("=" * 78)
print("Bisect: first checkpoint with low cosine = where divergence starts.")
print("  emb low  -> input/embedding/token-order differs (LB reorder, pos emb)")
print("  emb ~1, blk0 low -> attention or MLP in block 0")
print("  blk0 ~1, lnf low -> a later block or final LN")
print("If per-token cos is high for SOME tokens but ~0 for others -> token-order")
print("mismatch (HeadTail LB reorders even at ws=1), not a value bug.")
