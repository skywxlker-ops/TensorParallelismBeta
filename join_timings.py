import pandas as pd

# Load profiles
df_ours = pd.read_csv('DTensor/fullTPnoattn10steps nsys componentwise.csv')
df_mega = pd.read_csv('Megatron-LM/opt_cuda_gpu_kern_sum_base.csv')

df_ours.columns = df_ours.columns.str.strip()
df_mega.columns = df_mega.columns.str.strip()

# Sort by Total Time descending
df_ours = df_ours.sort_values(by='Total Time (ns)', ascending=False).reset_index(drop=True)
df_mega = df_mega.sort_values(by='Total Time (ns)', ascending=False).reset_index(drop=True)

# Select relevant columns and rename for clarity
df_ours_slim = df_ours[['Name', 'Total Time (ns)', 'Avg (ns)', 'Instances']].copy()
df_ours_slim['Our Total ms'] = df_ours_slim['Total Time (ns)'] / 1e6
df_ours_slim['Our Avg ms'] = df_ours_slim['Avg (ns)'] / 1e6
df_ours_slim = df_ours_slim.rename(columns={'Name': 'Our Kernel', 'Instances': 'Our Instances'})
df_ours_slim = df_ours_slim[['Our Kernel', 'Our Total ms', 'Our Avg ms', 'Our Instances']]

df_mega_slim = df_mega[['Name', 'Total Time (ns)', 'Avg (ns)', 'Instances']].copy()
df_mega_slim['Mega Total ms'] = df_mega_slim['Total Time (ns)'] / 1e6
df_mega_slim['Mega Avg ms'] = df_mega_slim['Avg (ns)'] / 1e6
df_mega_slim = df_mega_slim.rename(columns={'Name': 'Megatron Kernel', 'Instances': 'Mega Instances'})
df_mega_slim = df_mega_slim[['Megatron Kernel', 'Mega Total ms', 'Mega Avg ms', 'Mega Instances']]

# Concat side by side (outer join on index)
merged = pd.concat([df_mega_slim, df_ours_slim], axis=1)

# Format floats to 2 decimal places
for col in merged.columns:
    if 'ms' in col:
        merged[col] = merged[col].round(2)

merged.to_csv('kernel_timings_side_by_side.csv', index=False)
