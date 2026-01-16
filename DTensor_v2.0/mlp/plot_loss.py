#!/usr/bin/env python3
"""
Plot DTensor training loss from CSV log.
Usage: python3 plot_loss.py [training_log.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_loss(csv_path="training_log.csv", output_path="loss_vs_step.png"):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss
    ax.plot(df['step'], df['loss'], color='#4F46E5', linewidth=1.5, alpha=0.8)
    
    # Logarithmic scale for better visualization
    # ax.set_yscale('log')
    
    # Labels and title
    ax.set_title('DTensor GPT Training: Loss vs Step ', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss ', fontsize=12)
    
    # Grid for both major and minor ticks
    ax.grid(True, alpha=0.3, which='both')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also show if running interactively
    plt.show()

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "training_log.csv"
    plot_loss(csv_file)
