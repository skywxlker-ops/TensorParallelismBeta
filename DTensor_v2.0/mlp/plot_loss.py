#!/usr/bin/env python3
"""
Plot DTensor training loss from CSV log with moving average trend.
Usage: python3 plot_loss.py [training_log.csv]

Shows:
- Raw loss values (all fluctuations)
- Moving average trend (smoothed)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_loss(csv_path="training_log.csv", output_path="loss_vs_step.png"):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with subplot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. Plot raw loss with some transparency (actual fluctuations)
    ax.plot(df['step'], df['loss'], color="#0B68EB", linewidth=1, alpha=0.5, label='Raw Loss')
    
    # 2. Calculate and plot moving average (smoothed trend)
    window_size = max(50, len(df) // 20)  # Adaptive window (5% of data or 50, whichever is larger)
    if len(df) >= window_size:
        moving_avg = df['loss'].rolling(window=window_size, center=True).mean()
        ax.plot(df['step'], moving_avg, color="#E93010", linewidth=2.5, 
                label=f'Moving Average (window={window_size})', zorder=3)
    
    # Labels and title
    ax.set_title('DTensor GPT Training: Loss vs Step (with Moving Average)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.5)
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
    
    # Add statistics text box
    final_loss = df['loss'].iloc[-1]
    initial_loss = df['loss'].iloc[0]
    min_loss = df['loss'].min()
    avg_loss = df['loss'].mean()
    
    # stats_text = f'Initial: {initial_loss:.2f}\n'
    # stats_text += f'Final: {final_loss:.2f}\n'
    # stats_text += f'Min: {min_loss:.2f}\n'
    # stats_text += f'Avg: {avg_loss:.2f}'
    
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #         fontsize=9, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"  - Window size for moving average: {window_size}")
    print(f"  - Loss reduction: {initial_loss:.2f} → {final_loss:.2f} ({((final_loss - initial_loss) / initial_loss * 100):.1f}%)")
    
    # Also show if running interactively
    plt.show()

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "training_log.csv"
    plot_loss(csv_file)
