import pandas as pd

# Load the raw timing data
df_py = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Pytorch/pytorch_cp_timing_cuda_gpu_kern_sum_base.csv')
df_our = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/TrainingScripts/our_cp_timing_cuda_gpu_kern_sum_base.csv')

def categorize_kernel(name):
    name = str(name).lower()
    if 'flash_attn' in name:
        return 'Attention Core (Flash)'
    if 'softmax' in name:
        return 'Attention Core (Softmax)'
    if 'sgemm' in name or 'cutlass' in name or 'gemm' in name:
        return 'MatMul (Linear/MLP/QKV)'
    if 'nccl' in name:
        return 'Communication (Ring)'
    if 'elementwise' in name:
        return 'Elementwise Op (GeLU/Add/Loss)'
    if 'reduce' in name:
        return 'Reduction (LayerNorm/Loss/Merge)'
    if 'kernel2' in name:
        return 'Misc/Generated Kernel (Fused AdamW / LayerNorm)'
    return 'Other'

df_py['Category'] = df_py['Name'].apply(categorize_kernel)
df_our['Category'] = df_our['Name'].apply(categorize_kernel)

# Aggregate by Category
agg_py = df_py.groupby('Category')['Total Time (ns)'].sum().reset_index()
agg_py.rename(columns={'Total Time (ns)': 'PyTorch Total Config Time (ns)'}, inplace=True)

agg_our = df_our.groupby('Category')['Total Time (ns)'].sum().reset_index()
agg_our.rename(columns={'Total Time (ns)': 'Our Total Config Time (ns)'}, inplace=True)

merged = pd.merge(agg_py, agg_our, on='Category', how='outer').fillna(0)

# Also merge Instances
inst_py = df_py.groupby('Category')['Instances'].sum().reset_index()
inst_our = df_our.groupby('Category')['Instances'].sum().reset_index()

merged['PyTorch Instances'] = merged['Category'].map(inst_py.set_index('Category')['Instances']).fillna(0)
merged['Our Instances'] = merged['Category'].map(inst_our.set_index('Category')['Instances']).fillna(0)

# Sort by the maximum time taking component in PyTorch
merged = merged.sort_values(by='PyTorch Total Config Time (ns)', ascending=False)

# Convert ns to ms for readability
merged['PyTorch Time (ms)'] = (merged['PyTorch Total Config Time (ns)'] / 1e6).round(2)
merged['Our Time (ms)'] = (merged['Our Total Config Time (ns)'] / 1e6).round(2)

cols_to_show = ['Category', 'PyTorch Time (ms)', 'Our Time (ms)', 'PyTorch Instances', 'Our Instances']

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/compare_functionality.md', 'w') as f:
    f.write("# Functional Mapping Comparison\n\n")
    f.write("By grouping the kernels by the functionality they provide (instead of exact C++ symbol), we get a clearer picture of where time is spent.\n\n")
    f.write("| " + " | ".join(cols_to_show) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols_to_show)) + " |\n")
    for _, row in merged.iterrows():
        row_str = " | ".join([f"{int(row[c])}" if "Instances" in c else str(row[c]) for c in cols_to_show])
        f.write(f"| {row_str} |\n")
        
    f.write("\n### Details on Mapping\n")
    f.write("- **Attention Core**: PyTorch uses `sdpa_with_lse` in purely FP32 math (resulting in `cunn_SoftMaxForward/Backward`), whereas C++ uses highly optimized `flash_attn_fwd_kernel` and `flash_attn_bwd_dkdv_kernel`.\n")
    f.write("- **MatMul (Linear/MLP/QKV)**: Both use underlying hardware Matrix-Multiplies (`ampere_sgemm...`). *Note:* in PyTorch's pure FP32 manual Attention implementation, the Q*K and P*V multiplications also map to `sgemm`, heavily padding its Matmul time.\n")
    f.write("- **Communication**: `ncclDevKernel_SendRecv` is used by both implementations for the Context Parallel ring permutation.\n")
    f.write("- **Misc/Generated Kernel**: `Kernel2` is traditionally the generated/fused JIT kernel for PyTorch (e.g. Optimizer step / Loss) or C++ (e.g. FusedAdamW).\n")

print('Summary artifact done!')
