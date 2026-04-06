import re

with open("DTensor/gpt2_cp_test/context_parallel/FusedSDPABackwardKernel.cu", "r") as f:
    text = f.read()

# Fix 1: Memory Aliasing
text = text.replace(
    "float* s_wr = s_d  + BWD_TC_BQ;              // [32xHP] writeback for dK/dV",
    "float* s_wr = s_B  + BWD_TC_BQ*BWD_TC_BK;    // [32xHP] disjoint from ds and P"
)

# Fix 2: Missing syncthreads
text = text.replace(
    "s_B[row*BWD_TC_BK + lane] = p * (s_B[row*BWD_TC_BK + lane] - s_d[row]) * scale; // Store ds for dQ/dK calculation\n            }\n        }\n\n        // 3. dQ Update",
    "s_B[row*BWD_TC_BK + lane] = p * (s_B[row*BWD_TC_BK + lane] - s_d[row]) * scale; // Store ds for dQ/dK calculation\n            }\n        }\n        __syncthreads();\n\n        // 3. dQ Update"
)

with open("DTensor/gpt2_cp_test/context_parallel/FusedSDPABackwardKernel.cu", "w") as f:
    f.write(text)

print("Applied fixes")
