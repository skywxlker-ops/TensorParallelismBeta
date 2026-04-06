import pandas as pd
import numpy as np

df_py = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Pytorch/pytorch_cp_timing_cuda_gpu_kern_sum_base.csv')
df_our = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/TrainingScripts/our_cp_timing_cuda_gpu_kern_sum_base.csv')

def categorize_pytorch(name):
    name = str(name).lower()
    if 'softmax' in name:
        return 'Attention Core Math'
    if 'sgemm' in name or 'cutlass' in name or 'gemm' in name:
        return 'Matrix Multiply (Linear / QKV / MLP)'
    if 'nccl' in name:
        return 'Context Parallel Comm (Ring Attention)'
    if 'elementwise' in name:
        return 'Activation / Elementwise (GeLU / Add)'
    if 'reduce' in name:
        return 'Reduction (LayerNorm / Probabilities)'
    if 'kernel2' in name:
        return 'Generated Kernel (Optimizer / Loss)'
    return 'Other / Fallback'

def categorize_our(name):
    name = str(name).lower()
    if 'flash_attn' in name:
        return 'Attention Core Math'
    if 'nccl' in name:
         return 'Context Parallel Comm (Ring Attention)'
    if 'sgemm' in name or 'gemm' in name:
         return 'Matrix Multiply (Linear / QKV / MLP)'
    if 'gelu' in name or 'add' in name or 'scale' in name:
        return 'Activation / Elementwise (GeLU / Add)'
    if 'vln' in name or 'normalize' in name or 'reduce' in name:
        return 'Reduction (LayerNorm / Probabilities)'
    if 'copy' in name or 'broadcast' in name:
        return 'Memory Layout / Copy (Contiguous / Broadcast)'
    if 'adam' in name or 'grad_norm' in name:
        return 'Optimizer & Grad Clipping'
    if 'kernel2' in name:
        return 'Massive Unidentified Kernel (`Kernel2`)'
    return 'Other / Fallback'

df_py['Component'] = df_py['Name'].apply(categorize_pytorch)
df_our['Component'] = df_our['Name'].apply(categorize_our)

def aggregate_group(group):
    total_time_ms = group['Total Time (ns)'].sum() / 1e6
    instances = group['Instances'].sum()
    avg_ms = (total_time_ms / instances) if instances > 0 else 0
    sorted_group = group.sort_values(by='Total Time (ns)', ascending=False)
    names = []
    for n in sorted_group['Name'].unique():
        base = str(n).split('<')[0].split('(')[0].strip()
        if base not in names:
            names.append(base)
    name_str = " + ".join(names[:3])
    if len(names) > 3:
        name_str += " + ..."
    return pd.Series({
        'Kernel Name': name_str,
        'Total Time (ms)': round(total_time_ms, 2),
        'Avg (ms)': round(avg_ms, 2),
        'Instances': int(instances)
    })

agg_py = df_py.groupby('Component').apply(aggregate_group, include_groups=False).reset_index()
agg_our = df_our.groupby('Component').apply(aggregate_group, include_groups=False).reset_index()

agg_py = agg_py.rename(columns={'Kernel Name': 'PyTorch Kernel Name', 'Total Time (ms)': 'PyTorch Total Time (ms)', 'Avg (ms)': 'PyTorch Avg (ms)', 'Instances': 'PyTorch Instances'})
agg_our = agg_our.rename(columns={'Kernel Name': 'DTensor Kernel Name', 'Total Time (ms)': 'DTensor Total Time (ms)', 'Avg (ms)': 'DTensor Avg (ms)', 'Instances': 'DTensor Instances'})

all_components = pd.DataFrame({'Component': list(set(agg_py['Component']).union(set(agg_our['Component'])))})
final_df = pd.merge(all_components, agg_py, on='Component', how='left').merge(agg_our, on='Component', how='left').fillna('-')
final_df['sort_col'] = final_df['PyTorch Total Time (ms)'].apply(lambda val: -1 if val == '-' else float(val))
final_df = final_df.sort_values(by='sort_col', ascending=False).drop(columns=['sort_col'])

output_file = '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/kernel_mapping_cp_generated.csv'
final_df.to_csv(output_file, index=False)
