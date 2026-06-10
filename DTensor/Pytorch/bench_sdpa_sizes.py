"""
Standalone SDPA kernel timing — apples-to-apples vs our fused_attn kernel.
Times F.scaled_dot_product_attention at [B=4, H=12, T, D=64] causal, FP32,
for T = 256 / 512 / 1024, per backend (EFFICIENT, FLASH, MATH if available).

Our kernel (cp_sdpa_compare_test microbench) at the same shapes:
  256 -> 0.196 ms , 512 -> 0.662 ms , 1024 -> 2.299 ms
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dev = torch.device("cuda", 0)
B, H, D = 4, 12, 64
sizes = [256, 512, 1024]
NWARM, NITERS = 10, 50

backends = {
    "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "FLASH": SDPBackend.FLASH_ATTENTION,
    "MATH": SDPBackend.MATH,
}

def time_call(fn):
    for _ in range(NWARM):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(NITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / NITERS  # ms

print(f"=== PyTorch SDPA timing  B={B} H={H} D={D} FP32 ===")
print(f"{'backend/mask':<20}{'T=256(ms)':>12}{'T=512(ms)':>12}{'T=1024(ms)':>12}")
for name, be in backends.items():
    for causal in (True, False):
        tag = f"{name} {'causal' if causal else 'full'}"
        row = f"{tag:<20}"
        for T in sizes:
            q = torch.randn(B, H, T, D, device=dev, dtype=torch.float32)
            k = torch.randn(B, H, T, D, device=dev, dtype=torch.float32)
            v = torch.randn(B, H, T, D, device=dev, dtype=torch.float32)
            try:
                with sdpa_kernel([be]):
                    ms = time_call(lambda: F.scaled_dot_product_attention(
                        q, k, v, is_causal=causal))
                row += f"{ms:>12.4f}"
            except Exception:
                row += f"{'n/a':>12}"
        print(row)

print("\nOurs (fused_attn) causal:   256->0.196   512->0.662   1024->2.299")
print("If causal-time ~= full-time -> kernel computes the full square then masks")
print("(wastes ~half the work in causal mode).")
