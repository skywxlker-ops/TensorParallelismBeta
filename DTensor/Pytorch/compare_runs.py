#with sync hook method nsys
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
try:
    df_old = pd.read_csv('/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/TP_MLP_Training_logs/TP_MLP_Training_log4.csv')
    df_new = pd.read_csv('/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/gpt2_tp_test/TP_MLP_Torch_Logs/Pytorch_TP_log.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

if not df_old.empty:
    df_old.columns = df_old.columns.str.strip()
    df_new.columns = df_new.columns.str.strip()

    # Rename columns in df_new to match df_old
    rename_map = {
        'time_data': 'timer_data',
        'time_fwd': 'timer_fwd',
        'time_loss': 'timer_loss',
        'time_bwd': 'timer_bwd',
        'time_clip': 'timer_clip',
        'time_optim': 'timer_optim',
        't_tok_emb': 'timer_tok_emb',
        't_pos_emb': 'timer_pos_emb',
        't_mlp': 'timer_mlp',
        't_ln_f': 'timer_ln_f',
        't_lm_head': 'timer_lm_head',
        'norm': 'grad_norm'
    }
    df_new.rename(columns=rename_map, inplace=True)

# 1. Get the current progress (max step of the new run)
current_max_step = df_new['step'].max()

# 2. Filter the old run to match the current progress
df_old_truncated = df_old[df_old['step'] <= current_max_step]

# 3. Plotting
plt.figure(figsize=(10, 6))

plt.plot(df_new['step'], df_new['loss'], label='Pytorch 2 GPU Run', linewidth=2)
plt.plot(df_old_truncated['step'], df_old_truncated['loss'], label='Old Synchronization Run', linestyle = "--", color = "red", alpha=0.5)

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title(f'Training Loss  (Up to Step {current_max_step})')
plt.legend()
plt.grid(True, linestyle='-', alpha=0.6)

plt.savefig('comparison_loss.png')
# plt.show()


plt.figure(figsize=(10, 6)) # Create a new figure for the second plot
plt.plot(df_new['step'], df_new['tok_per_sec'], label='Pytorch 2 GPU Run', linewidth=2)
plt.plot(df_old_truncated['step'], df_old_truncated['tok_per_sec'], label='Old Synchronization Run', linestyle = "--", color = "red", alpha=0.5)

plt.xlabel('Step')
plt.ylabel('Throughput')
plt.title(f'Training Thoughput (Up to Step {current_max_step})')
plt.legend()
plt.grid(True, linestyle='-', alpha=0.6)

plt.savefig('comparison_throughput.png')
# plt.show()

print("Max. Throughput new:",df_new['tok_per_sec'].max())
print("Avg. Throughput new:",df_new['tok_per_sec'].mean())
print("Min Train Loss new:",df_new['loss'][:].min())
# Handle potentially missing val_loss or different structure
if 'val_loss' in df_new.columns:
    print("Min Val Loss new:",df_new['val_loss'].where(df_new['val_loss']!=(-1)).min())
else:
    print("Min Val Loss new: N/A")

# print("Avg. Time new:",df_new['dt_ms'].mean()) # commented out in original
print("Avg. Time new:",df_new['dt_ms'].mean())

for col in ['timer_data', 'timer_fwd', 'timer_loss', 'timer_bwd', 'timer_clip', 'timer_optim', 
            'timer_tok_emb', 'timer_pos_emb', 'timer_mlp', 'timer_ln_f', 'timer_lm_head']:
    if col in df_new.columns:
        print(f"Avg. {col} new:", df_new[col].mean())

print("\nMax. Throughput old:",df_old_truncated['tok_per_sec'].max())
print("Avg. Throughput old:",df_old_truncated['tok_per_sec'].mean())
print("Min Train Loss old:",df_old_truncated['loss'][:].min())
if 'val_loss' in df_old_truncated.columns:
    print("Min Val Loss old:",df_old_truncated['val_loss'].where(df_old_truncated['val_loss']!=(-1)).min())
else:
    print("Min Val Loss old: N/A")

print("Avg. Time old:",df_old_truncated['dt_ms'].mean())

# Print mean times if columns exist
for col in ['timer_data', 'timer_fwd', 'timer_loss', 'timer_bwd', 'timer_clip', 'timer_optim', 
            'timer_tok_emb', 'timer_pos_emb', 'timer_mlp', 'timer_ln_f', 'timer_lm_head']:
    if col in df_old_truncated.columns:
        print(f"Avg. {col} old:", df_old_truncated[col].mean())
