import pandas as pd
df = pd.read_csv('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/compare_pytorch_our_kernels.csv')
cols = ['Name', 'Pytorch Total Time (ns)', 'Our Total Time (ns)', 'Pytorch Avg (ns)', 'Our Avg (ns)', 'Speedup (Pytorch/Our)']
df_subset = df[cols].head(30)
with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/compare_kernels_results.md', 'w') as f:
    f.write("# Kernel Timing Comparison (Top 30 by PyTorch Time)\n\n")
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for _, row in df_subset.iterrows():
        row_str = " | ".join([str(x) for x in row.values])
        f.write(f"| {row_str} |\n")
print('Done!')
