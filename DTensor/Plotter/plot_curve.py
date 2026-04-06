import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

def plot_loss(csv_path="training_log.csv", output_path="loss_vs_step.png"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Handle existing output path by adding a counter
    if os.path.exists(output_path):
        base, ext = os.path.splitext(output_path)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        output_path = f"{base}_{counter}{ext}"

    # Load data
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print(f"Warning: {csv_path} is empty.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss
    ax.plot(df['global_step'], df['loss_value'], color="#1EC7ED", linewidth = 2.0, alpha= 1.0)
    
    # Labels and title
    ax.set_title('DTensor GPT Training: Loss vs Step ', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    
    # Grid for both major and minor ticks
    ax.grid(True, alpha=0.7, which='both')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "training_log.csv"
    plot_loss(csv_file)