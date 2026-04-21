import pandas as pd
import numpy as np
import os

# File paths
OUR_CSV = '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/opt_gpt2_cp_62k.csv'
PYTORCH_CSV = '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Pytorch/opt_pytorch_cp_64k.csv'
OUTPUT_CSV = '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/combined_nsys_comparison.csv'

# Load data
df_our = pd.read_csv(OUR_CSV)
df_py = pd.read_csv(PYTORCH_CSV)

def map_kernel_to_component(name, is_pytorch):
    name = str(name)
    name_l = name.lower()
    
    # 1. Matrix Multiply (MLP, QKV, LM Head)
    if 'sgemm' in name_l or 'gemm' in name_l or 'cutlass' in name_l:
        if 'fmha' in name_l: # Flash Attention uses CUTLASS
            pass 
        else:
            return 'Matrix Multiply (Linear / MLP / LM Head)'
    
    if 'kernel2' in name_l:
        # In this project, Kernel2 often represents CuBLAS Lt or fused GEMMs
        return 'Matrix Multiply (Linear / MLP / LM Head)'
            
    # 2. Attention
    if 'fused_attn' in name_l or 'fmha_cutlassf' in name_l:
        return 'Attention (Forward)'
    if 'mem_efficient_bwd' in name_l or 'fmha_cutlassb' in name_l:
        return 'Attention (Backward)'
        
    # 3. Communications
    if 'nccl' in name_l:
        if 'sendrecv' in name_l:
            return 'Context Parallel Comm (Ring SendRecv)'
        elif 'allreduce' in name_l:
            return 'AllReduce (NCCL)'
        else:
            return 'Other NCCL'

    # 4. LayerNorm
    if 'vln_fwd' in name_l or 'vectorized_layer_norm' in name_l:
        return 'LayerNorm (Forward)'
    if 'vln_bwd_input' in name_l or 'layer_norm_grad_input' in name_l:
        return 'LayerNorm (Backward Input)'
    if 'vln_bwd_gamma_beta' in name_l or 'gammabetabackward' in name_l:
        return 'LayerNorm (Backward Gamma/Beta)'

    # 5. Cross Entropy / Loss (SCELoss)
    if ('sparse_ce' in name_l or 'sparsecenormalize' in name_l 
            or 'sparsecereduce' in name_l or 'nll_loss' in name_l 
            or 'cunn_softmax' in name_l):
        return 'Cross Entropy / Loss (SCELoss)'

    # 6. Embeddings
    if 'embedding_forward' in name_l or 'vectorized_gather' in name_l:
        return 'Embedding (Forward)'
    if 'embedding_backward' in name_l:
        return 'Embedding (Backward)'
    
    # 7. Optimizers & Clipping
    if 'adam' in name_l or (is_pytorch and 'multi_tensor_apply' in name_l):
        return 'Adam Optimizer (Step)'
    if 'grad_norm' in name_l or 'lpnorm' in name_l:
        return 'Gradient Clipping (Norm L2)'
    if 'compute_clip_coef' in name_l or (not is_pytorch and 'multi_tensor_scale' in name_l):
        return 'Gradient Clipping (Scale)'

    # 8. Add / Residuals
    if 'add_kernel' in name_l:
        return 'Add / Residuals (Forward)'

    # 9. GeLU / Elementwise
    if 'gelu' in name_l:
        if 'backward' in name_l:
             return 'GeLU Activation (Backward)'
        return 'GeLU Activation (Forward)'
    
    if is_pytorch and 'elementwise' in name_l:
        return 'Generic Elementwise / GeLU / Add'
        
    if 'vectorized_kernel_impl' in name_l or 'broadcast_scale' in name_l or 'mul_kernel' in name_l:
        return 'Broadcasting / Scaling / Misc Elementwise'
        
    if ('sub_kernel' in name_l or 'k_div' in name_l or 'broadcast' in name_l 
            or 'convert_type' in name_l or 'unary_kernel' in name_l or 'scalar_div' in name_l):
        return 'Broadcasting / Scaling / Misc Elementwise'

    # 10. Memops / Utils
    if 'copy' in name_l:
        return 'Data Layout / Strided Copy'
    if 'gen_sequenced' in name_l:
        return 'Sequence Generation / Masks'
    if 'generate_seed' in name_l or 'generate_pseudo' in name_l:
        return 'Pseudorandom Number Generation'

    # 11. Reductions
    if 'reduce' in name_l:
        if 'splitk' in name_l:
            return 'Matrix Multiply (Accumulation/SplitK)'
        return 'Generic Reduction'

    return f"Misc: {name.split('<')[0].split('(')[0]}"

# Apply mapping
df_our['Component'] = df_our['Name'].apply(lambda x: map_kernel_to_component(x, False))
df_py['Component'] = df_py['Name'].apply(lambda x: map_kernel_to_component(x, True))

def aggregate_by_component(df, prefix):
    # Ensure columns are numeric
    df['Total Time (ns)'] = pd.to_numeric(df['Total Time (ns)'], errors='coerce')
    df['Instances'] = pd.to_numeric(df['Instances'], errors='coerce')
    
    def agg_func(group):
        total_time_ns = group['Total Time (ns)'].sum()
        instances = group['Instances'].sum()
        avg_ns = total_time_ns / instances if instances > 0 else 0
        
        # Sort kernels by total time within group
        sorted_kernels = group.sort_values('Total Time (ns)', ascending=False)['Name'].unique()
        kernel_names = " + ".join(str(k).split('<')[0].split('(')[0] for k in sorted_kernels[:3])
        if len(sorted_kernels) > 3:
            kernel_names += " + ..."
            
        return pd.Series({
            f'{prefix} Kernel Name': kernel_names,
            f'{prefix} Total Time (ms)': total_time_ns / 1e6,
            f'{prefix} Avg (ms)': avg_ns / 1e6,
            f'{prefix} Instances': instances
        })

    return df.groupby('Component').apply(agg_func, include_groups=False).reset_index()

# Aggregate
agg_our = aggregate_by_component(df_our, 'DTensor')
agg_py = aggregate_by_component(df_py, 'PyTorch')

# Combine
all_components = sorted(list(set(agg_our['Component']).union(set(agg_py['Component']))))
final_df = pd.DataFrame({'Component': all_components})
final_df = final_df.merge(agg_py, on='Component', how='left').merge(agg_our, on='Component', how='left').fillna('-')

# Sort by max total time
def sort_key(x):
    if x == '-': return -1.0
    try: return float(x)
    except: return -1.0

final_df['sort_val'] = final_df.apply(lambda row: max(sort_key(row['PyTorch Total Time (ms)']), sort_key(row['DTensor Total Time (ms)'])), axis=1)
final_df = final_df.sort_values('sort_val', ascending=False).drop(columns=['sort_val'])

# Formatting
for col in final_df.columns:
    if 'Time (ms)' in col or 'Avg (ms)' in col:
        final_df[col] = final_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    if 'Instances' in col:
        final_df[col] = final_df[col].apply(lambda x: str(int(x)) if isinstance(x, (int, float)) else x)

# Save
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Combined report saved to {OUTPUT_CSV}")
