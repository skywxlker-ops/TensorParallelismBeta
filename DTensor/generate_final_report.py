import pandas as pd

df_py = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Pytorch/pytorch_cp_timing_cuda_gpu_kern_sum_base.csv')
df_our = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/TrainingScripts/our_cp_timing_cuda_gpu_kern_sum_base.csv')

def categorize_pytorch_kernel(name):
    name = str(name).lower()
    if 'flash_attn' in name:
        return 'Attention Base Computation'
    if 'softmax' in name:
        return 'Attention Core Math (Online Softmax)'
    if 'sgemm' in name or 'cutlass' in name or 'gemm' in name:
        return 'Dense Matrix Multiplication (Linear/QKV/MLP)'
    if 'nccl' in name:
        return 'Context Parallel Communication (Ring Attention)'
    if 'elementwise' in name:
        return 'Elementwise Op (GeLU / Add / Scale)'
    if 'reduce' in name:
        return 'Reduction Op (LayerNorm / CrossEntropy / Sum)'
    if 'kernel2' in name:
        return 'Generated Kernel (Optimizer Step / Loss)'
    return 'Other'

def categorize_our_kernel(name):
    name = str(name).lower()
    if 'flash_attn' in name:
        return 'Attention Base Computation'
    if 'nccl' in name:
         return 'Context Parallel Communication (Ring Attention)'
    if 'sgemm' in name or 'gemm' in name:
         return 'Dense Matrix Multiplication (Linear/QKV/MLP)'
    if 'add' in name or 'gelu' in name or 'copy' in name or 'scale' in name:
        return 'Elementwise Op (GeLU / Add / Scale)'
    if 'reduce' in name or 'normalize' in name or 'grad_norm' in name:
        return 'Reduction Op (LayerNorm / CrossEntropy / Sum)'
    if 'adam' in name:
        return 'Generated Kernel (Optimizer Step / Loss)'
    if 'kernel2' in name:
        # Based on its high instance count and lack of GEMMs, Kernel2 might be their naive MatMul 
        # or massive elementwise blocks. We will map it generally.
        return 'Custom/Generated C++ Kernels (`Kernel2`)'
    return 'Other'

df_py['Category'] = df_py['Name'].apply(categorize_pytorch_kernel)
df_our['Category'] = df_our['Name'].apply(categorize_our_kernel)

agg_py = df_py.groupby('Category')[['Total Time (ns)', 'Instances']].sum().reset_index()
agg_our = df_our.groupby('Category')[['Total Time (ns)', 'Instances']].sum().reset_index()

merged = pd.merge(agg_py, agg_our, on='Category', how='outer', suffixes=(' PyTorch', ' Our C++')).fillna(0)

# Sort based on PyTorch time
merged = merged.sort_values(by='Total Time (ns) PyTorch', ascending=False)

def to_ms(ns):
    return f"{(ns / 1e6):.2f}"

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/compare_functionality_detailed.md', 'w') as f:
    f.write("# Kernel Functionality Mapping between PyTorch and C++ Code\n\n")
    
    f.write("I have mapped the kernels from both performance profiles (`Pytorch/...` and `TrainingScripts/...`) strictly to the **functional components** they execute in their respective scripts (`gpt2_context_parallel_fp32.py` and `gpt2_cp_test.cpp`).\n\n")
    
    f.write("> [!WARNING]\n")
    f.write("> **Missing CuBLAS and NCCL in C++ Trace**: The custom codebase trace `our_cp_timing_cuda_gpu_kern_sum_base.csv` does **not** contain any occurrences of `sgemm`, `gemm`, or `nccl` kernels. This means either cuBLAS/NCCL wasn't tracked properly by Nsys during the execution run, or the linear operations are mapping directly into the massive un-named 913ms `Kernel2` block.\n\n")
    
    f.write("| Functionality Category | PyTorch Time (ms) | PyTorch Count | Our C++ Time (ms) | Our C++ Count |\n")
    f.write("|---|---|---|---|---|\n")
    
    for _, row in merged.iterrows():
        cat = row['Category']
        pty = to_ms(row['Total Time (ns) PyTorch'])
        ptc = int(row['Instances PyTorch'])
        oty = to_ms(row['Total Time (ns) Our C++'])
        otc = int(row['Instances Our C++'])
        f.write(f"| {cat} | {pty} | {ptc} | {oty} | {otc} |\n")
        
    f.write("\n### Deep Dive into the Code Functionality\n\n")
    f.write("1. **Dense Matrix Multiplication** (`nn.Linear` layers for `c_attn`, `c_proj`, `fc_up`, `fc_down` and LM Head):\n")
    f.write("   - **PyTorch**: Native MatMuls map cleanly to `ampere_sgemm_128x64_tn` and similar kernels. Additionally, because PyTorch is running a manual FP32 `sdpa_with_lse()` implementation, the underlying $Q \\times K^T$ and $P \\times V$ operations in the Ring Attention step are *also* handled via standard `sgemm`.\n")
    f.write("   - **Our C++**: No generic `sgemm` kernels were logged. The linear layers either fallback to your custom `Kernel2` implementation natively inside `TensorLib`, or the trace missed them.\n\n")
    f.write("2. **Attention Core Math**:\n")
    f.write("   - **PyTorch**: Executes the math explicitly via `torch.softmax()` inside the loop -> `cunn_SoftMaxForward` and `cunn_SoftMaxBackward`.\n")
    f.write("   - **Our C++**: Replaces the generic primitive multiplications and softmaxes directly with your monolithic kernels: `flash_attn_fwd_kernel`, `flash_attn_bwd_dkdv_kernel`, and `flash_attn_bwd_dq_kernel` (called in `ContextParallel::forward_cp`).\n\n")
    f.write("3. **Context Parallel Communication**:\n")
    f.write("   - **PyTorch**: Evaluates via `torch.distributed.batch_isend_irecv` -> Maps correctly to `ncclDevKernel_SendRecv`.\n")
    f.write("   - **Our C++**: Uses raw NCCL operations in `ContextParallel.h` but again, `ncclDevKernel_SendRecv` simply isn't in your `our_cp_timing...` CSV. Check if you profiled with NCCL tracing enabled!\n\n")
    f.write("4. **Activation Functions / Normalizations** (`LayerNorm`, `GeLU`):\n")
    f.write("   - **PyTorch**: Computes via composite `vectorized_elementwise_kernel`.\n")
    f.write("   - **Our C++**: Has explicit `fused_gelu_kernel`, `sparseCENormalize_kernel`, `vln_fwd_f32_kernel` (Vectorized Layer Norm), etc., logged correctly.\n")

print("Generated mapping comparison.")
