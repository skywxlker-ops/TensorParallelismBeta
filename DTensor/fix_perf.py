import re

with open("DTensor/gpt2_cp_test/context_parallel/FusedSDPABackwardKernel.cu", "r") as f:
    text = f.read()

stagger_loop = """
    // Main loop over KV tiles (Staggered to prevent atomic collision on dK/dV)
    const int num_k_steps = (max_kj + BWD_TC_BK - 1) / BWD_TC_BK;
    const int start_step = (blockIdx.x * 7) % max(1, num_k_steps); // Pseudo-random stagger

    for (int step_idx = 0; step_idx < num_k_steps; ++step_idx) {
        int64_t kj_blk = ((start_step + step_idx) % num_k_steps) * BWD_TC_BK;
"""

text = text.replace(
    "    // Main loop over KV tiles\n    for (int64_t kj_blk = 0; kj_blk < max_kj; kj_blk += BWD_TC_BK) {",
    stagger_loop
)

with open("DTensor/gpt2_cp_test/context_parallel/FusedSDPABackwardKernel.cu", "w") as f:
    f.write(text)

print("Applied staggering loop for performance.")
