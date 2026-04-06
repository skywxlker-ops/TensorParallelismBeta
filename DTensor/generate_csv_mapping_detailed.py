import pandas as pd

df_py = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Pytorch/pytorch_cp_timing_cuda_gpu_kern_sum_base.csv')
df_our = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/TrainingScripts/our_cp_timing_cuda_gpu_kern_sum_base.csv')

def map_kernel_exact(name, is_pytorch):
    name = str(name)
    name_l = name.lower()
    
    # 1. MatMuls
    if 'sgemm' in name_l or 'gemm' in name_l or 'cutlass' in name_l:
        return 'Matrix Multiply (Linear / QKV / MLP)'
    # DTensor CuBLAS LtMatmul surfaces as Kernel2; PyTorch Kernel2 is unidentified
    if 'kernel2' in name_l:
        if not is_pytorch:
            return 'Matrix Multiply (Linear / QKV / MLP)'
        return 'Unidentified / Fused Generated Kernel (`Kernel2`)'
    # splitKreduce is a split-K GEMM accumulation pass, not a loss reduction
    if 'splitkreduce' in name_l:
        return 'Matrix Multiply (Linear / QKV / MLP)'
        
    # 2. Attention
    # softmax_warp_forward is part of PyTorch's standard attention path
    if 'flash_attn_fwd' in name_l or 'cunn_softmaxforward' in name_l or 'softmax_warp_forward' in name_l:
        return 'Attention Computation (Forward)'
    if 'flash_attn_bwd' in name_l or 'cunn_softmaxbackward' in name_l or 'softmax_warp_backward' in name_l:
        return 'Attention Computation (Backward)'

    # 3. Communications
    if 'nccl' in name_l:
        if 'sendrecv' in name_l:
            return 'Context Parallel Comm (Ring SendRecv)'
        else:
            return 'Context Parallel Comm (Ring AllGather/AllReduce)'

    # 4. LayerNorm (before generic elementwise so vln_* doesn't fall through)
    if 'vln_fwd' in name_l or 'vectorized_layer_norm' in name_l:
        return 'LayerNorm (Forward)'
    if 'vln_bwd_input' in name_l or 'layer_norm_grad_input' in name_l:
        return 'LayerNorm (Backward Input)'
    if 'vln_bwd_gamma_beta' in name_l or 'gammabetabackward' in name_l:
        return 'LayerNorm (Backward Gamma/Beta)'

    # 5. Cross Entropy — all CE kernels from both sides map to one component per phase
    # PyTorch: nll_loss_forward + nll_loss_backward + some reduce
    # DTensor: sparse_ce_forward + sparseCEReduce + sparseCENormalize
    if ('sparse_ce_forward' in name_l or 'sparsecenormalize' in name_l
            or 'nll_loss_forward' in name_l or 'sparsecereduce' in name_l
            or 'nll_loss_backward' in name_l):
        return 'Cross Entropy'

    # 6. Embeddings
    if 'embedding_forward' in name_l or 'vectorized_gather' in name_l:
        return 'Embedding (Forward)'
    if 'embedding_backward' in name_l:
        return 'Embedding (Backward)'
    # PyTorch embedding backward decomposes into sort + scatter + partial-segment passes
    if ('deviceradixsort' in name_l or 'deviceunique' in name_l
            or 'devicescan' in name_l or 'devicecompact' in name_l
            or 'sum_and_scatter' in name_l or 'compute_grad_weight' in name_l
            or 'krn_partial_segment' in name_l or 'krn_partials_per' in name_l
            or 'compute_num_of_partial' in name_l):
        return 'Embedding (Backward)'

    # 7. Optimizers & Clipping
    if 'adam' in name_l or (is_pytorch and 'multi_tensor_apply' in name_l):
        return 'Adam Optimizer (Step)'
    if 'grad_norm' in name_l or 'lpnorm' in name_l:
        return 'Gradient Clipping (Norm L2)'
    if 'compute_clip_coef' in name_l or (not is_pytorch and 'multi_tensor_scale' in name_l):
        return 'Gradient Clipping (Scale)'

    # 8. Add / Residuals
    if 'add' in name_l and 'broadcast' in name_l:
        return 'Add / Residuals (Inplace/Basic)'
    if 'add' in name_l:
        return 'Add / Residuals (Inplace/Basic)'

    # 9. Elementwise / Activations / Misc
    # PyTorch fuses GeLU + activations + type ops into generic elementwise_kernel.
    # DTensor replaces these with purpose-built kernels; map both sides to one row.
    if 'gelu' in name_l or 'sigmoid' in name_l:
        return 'Elementwise / Activations (GeLU, Sigmoid)'
    if is_pytorch and 'elementwise' in name_l and 'unrolled' not in name_l:
        return 'Elementwise / Activations (GeLU, Sigmoid)'
    if is_pytorch and 'elementwise' in name_l and 'unrolled' in name_l:
        return 'Add / Residuals (Inplace/Basic)'
    # DTensor-side misc elementwise ops that correspond to PyTorch generic elementwise
    if ('sub_kernel' in name_l or 'mul_kernel' in name_l or 'k_div' in name_l
            or 'broadcast' in name_l or 'convert_type' in name_l
            or 'unary_kernel' in name_l or 'sigmoid' in name_l):
        return 'Elementwise / Activations (GeLU, Sigmoid)'

    # 10. Memops / Utils
    if 'copy' in name_l:
        return 'Data Layout / Strided Copy'
    if 'gen_sequenced' in name_l:
        return 'Sequence Generation / Masks'
    if 'generate_seed' in name_l:
        return 'Pseudorandom Number Generation / Seed'

    # Reduction fallback
    if 'reduce' in name_l:
        return 'Generic Reduction (Loss / Sum)'

    return f"Specific: {name.split('<')[0].split('(')[0]}"

df_py['Component'] = df_py['Name'].apply(lambda x: map_kernel_exact(x, True))
df_our['Component'] = df_our['Name'].apply(lambda x: map_kernel_exact(x, False))

def aggregate_func(group):
    total_ms = group['Total Time (ns)'].sum() / 1e6
    instances = group['Instances'].sum()
    avg_ms = (total_ms / instances) if instances > 0 else 0
    
    names = []
    for n in group.sort_values('Total Time (ns)', ascending=False)['Name'].unique():
        base = str(n).split('<')[0].split('(')[0].strip()
        if base not in names:
            names.append(base)
    # keep all for full transparency since they don't want anything hidden
    kernel_names = " + ".join(names[:5])
    if len(names) > 5:
        kernel_names += " + ..."
        
    return pd.Series({
        'Kernel Name': kernel_names,
        'Total Time (ms)': total_ms,
        'Avg (ms)': avg_ms,
        'Instances': instances
    })

agg_py = df_py.groupby('Component').apply(aggregate_func, include_groups=False).reset_index()
agg_our = df_our.groupby('Component').apply(aggregate_func, include_groups=False).reset_index()

agg_py = agg_py.rename(columns={'Kernel Name': 'PyTorch Kernel Name', 'Total Time (ms)': 'PyTorch Total Time (ms)', 'Avg (ms)': 'PyTorch Avg (ms)', 'Instances': 'PyTorch Instances'})
agg_our = agg_our.rename(columns={'Kernel Name': 'DTensor Kernel Name', 'Total Time (ms)': 'DTensor Total Time (ms)', 'Avg (ms)': 'DTensor Avg (ms)', 'Instances': 'DTensor Instances'})

all_components = pd.DataFrame({'Component': list(set(agg_py['Component']).union(set(agg_our['Component'])))})
final_df = pd.merge(all_components, agg_py, on='Component', how='left').merge(agg_our, on='Component', how='left').fillna('-')

def sort_key(val): return -1 if val == '-' else float(val)

# Custom sort order priority based on component name or time
final_df['sort_col'] = final_df['PyTorch Total Time (ms)'].apply(sort_key)
final_df = final_df.sort_values(by='sort_col', ascending=False).drop(columns=['sort_col'])

# Format numbers nicely
for col in final_df.columns:
    if 'ms)' in col or 'Instances' in col:
        final_df[col] = final_df[col].apply(lambda x: f"{float(x):.2f}" if x != '-' and '.' in str(float(x)) else x)
        if 'Instances' in col:
            final_df[col] = final_df[col].apply(lambda x: str(int(float(x))) if x != '-' else '-')

output_file = '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/kernel_mapping_all_detailed.csv'
final_df.to_csv(output_file, index=False)
print("Saved to", output_file)
