import pandas as pd

df_pytorch = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Pytorch/pytorch_cp_timing_cuda_gpu_kern_sum_base.csv')
df_our = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/TrainingScripts/our_cp_timing_cuda_gpu_kern_sum_base.csv')

df_pytorch = df_pytorch.rename(columns=lambda x: f"Pytorch {x}" if x != 'Name' else x)
df_our = df_our.rename(columns=lambda x: f"Our {x}" if x != 'Name' else x)

df_merged = pd.merge(df_pytorch, df_our, on='Name', how='outer')

# Fill NaN with 0 for time/instances, or empty string
df_merged = df_merged.fillna(0)

df_merged = df_merged.sort_values(by=['Pytorch Total Time (ns)', 'Our Total Time (ns)'], ascending=[False, False])

if 'Pytorch Avg (ns)' in df_merged.columns and 'Our Avg (ns)' in df_merged.columns:
    import numpy as np
    df_merged['Speedup (Pytorch/Our)'] = np.where(df_merged['Our Avg (ns)'] > 0, df_merged['Pytorch Avg (ns)'] / df_merged['Our Avg (ns)'], 0)
    df_merged['Speedup (Pytorch/Our)'] = df_merged['Speedup (Pytorch/Our)'].round(2)

output_csv = '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/compare_pytorch_our_kernels.csv'
df_merged.to_csv(output_csv, index=False)
print(f"Successfully generated merged comparison CSV at {output_csv}")
