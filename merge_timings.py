import pandas as pd
import numpy as np

# Load the mapping
mapping = pd.read_csv('kernel_mapping_detailed.csv')

# Load the profiles
df_ours = pd.read_csv('DTensor/fullTPnoattn10steps nsys componentwise.csv')
df_mega = pd.read_csv('Megatron-LM/opt_cuda_gpu_kern_sum_base.csv')

# Clean columns
df_ours.columns = df_ours.columns.str.strip()
df_mega.columns = df_mega.columns.str.strip()

# Create lookup dictionaries for Total Time (ms) and Avg (ms)
ours_total = dict(zip(df_ours['Name'], df_ours['Total Time (ns)'] / 1e6))
ours_avg = dict(zip(df_ours['Name'], df_ours['Avg (ns)'] / 1e6))
ours_inst = dict(zip(df_ours['Name'], df_ours['Instances']))

mega_total = dict(zip(df_mega['Name'], df_mega['Total Time (ns)'] / 1e6))
mega_avg = dict(zip(df_mega['Name'], df_mega['Avg (ns)'] / 1e6))
mega_inst = dict(zip(df_mega['Name'], df_mega['Instances']))

# We need to handle multiple kernels per row in the mapping
def safe_get(d, k):
    if pd.isna(k) or k == '-': return ''
    if k in d: return round(d[k], 3)
    return ''

mapping['Megatron Total Time (ms)'] = mapping['Megatron Kernel'].apply(lambda k: safe_get(mega_total, k))
mapping['Megatron Avg (ms)'] = mapping['Megatron Kernel'].apply(lambda k: safe_get(mega_avg, k))
mapping['Megatron Instances'] = mapping['Megatron Kernel'].apply(lambda k: safe_get(mega_inst, k))

mapping['C++ Total Time (ms)'] = mapping['C++ Kernel'].apply(lambda k: safe_get(ours_total, k))
mapping['C++ Avg (ms)'] = mapping['C++ Kernel'].apply(lambda k: safe_get(ours_avg, k))
mapping['C++ Instances'] = mapping['C++ Kernel'].apply(lambda k: safe_get(ours_inst, k))

# Overrides for multiple kernels combined
for idx, row in mapping.iterrows():
    if row['Megatron Kernel'] == 'compute_grad_weight + sort kernels':
        keys = ['compute_grad_weight', 'sum_and_scatter', 'DeviceRadixSortOnesweepKernel', 'splitKreduce_kernel', 'index_elementwise_kernel', 'DeviceRadixSortHistogramKernel', 'elementwise_kernel_with_index', 'DeviceUniqueByKeySweepKernel', 'krn_partial_segment_offset', 'DeviceScanKernel', 'krn_partials_per_segment', 'DeviceRadixSortExclusiveSumKernel', 'DeviceCompactInitKernel', 'compute_num_of_partial_segments', 'DeviceScanInitKernel']
        total = sum(mega_total.get(k, 0) for k in keys)
        mapping.at[idx, 'Megatron Total Time (ms)'] = round(total, 3)
        mapping.at[idx, 'Megatron Avg (ms)'] = ''
        mapping.at[idx, 'Megatron Instances'] = ''
        
    if row['Megatron Kernel'] == 'elementwise_kernel + reduce_kernel':
        total = mega_total.get('elementwise_kernel', 0) + mega_total.get('reduce_kernel', 0)
        mapping.at[idx, 'Megatron Total Time (ms)'] = round(total, 3)
        mapping.at[idx, 'Megatron Avg (ms)'] = ''
        mapping.at[idx, 'Megatron Instances'] = ''
        
    if row['C++ Kernel'] == 'add_inplace_kernel / add_kernel':
        total = ours_total.get('add_inplace_kernel', 0) + ours_total.get('add_kernel', 0)
        mapping.at[idx, 'C++ Total Time (ms)'] = round(total, 3)
        mapping.at[idx, 'C++ Avg (ms)'] = ''
        mapping.at[idx, 'C++ Instances'] = ''
        
    if row['C++ Kernel'] == 'add_inplace_kernel_broadcast / add_kernel_nd_broadcast':
        total = ours_total.get('add_inplace_kernel_broadcast', 0) + ours_total.get('add_kernel_nd_broadcast', 0)
        mapping.at[idx, 'C++ Total Time (ms)'] = round(total, 3)
        mapping.at[idx, 'C++ Avg (ms)'] = ''
        mapping.at[idx, 'C++ Instances'] = ''

mapping.to_csv('kernel_mapping_detailed_with_timings.csv', index=False)
