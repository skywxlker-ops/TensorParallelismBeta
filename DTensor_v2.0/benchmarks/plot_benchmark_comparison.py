#!/usr/bin/env python3
"""
Plot benchmark comparison: PyTorch DTensor vs Custom DTensor
Reads CSV files from both benchmarks and creates a comparison plot.

Usage:
    python plot_benchmark_comparison.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    pytorch_csv = os.path.join(script_dir, 'pytorch_benchmark_results.csv')
    dtensor_csv = os.path.join(script_dir, 'dtensor_benchmark_results.csv')
    
    # Check files exist
    if not os.path.exists(pytorch_csv):
        print(f"Error: {pytorch_csv} not found. Run pytorch_benchmarking_vary.py first.")
        return
    if not os.path.exists(dtensor_csv):
        print(f"Error: {dtensor_csv} not found. Run shard_benchmarking_vary first.")
        return
    
    # Load data
    pytorch_df = pd.read_csv(pytorch_csv)
    dtensor_df = pd.read_csv(dtensor_csv)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Size vs Time
    ax1 = axes[0]
    ax1.plot(pytorch_df['size_mb'], pytorch_df['time_ms'], 'o-', 
             label='PyTorch DTensor', markersize=6, linewidth=2, color='#1f77b4')
    ax1.plot(dtensor_df['size_mb'], dtensor_df['time_ms'], 's-', 
             label='Custom DTensor', markersize=6, linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Tensor Size (MB)', fontsize=12)
    ax1.set_ylabel('Sharding Time (ms)', fontsize=12)
    ax1.set_title('Tensor Size vs Sharding Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Size vs Throughput
    ax2 = axes[1]
    ax2.plot(pytorch_df['size_mb'], pytorch_df['throughput_mbs'], 'o-', 
             label='PyTorch DTensor', markersize=6, linewidth=2, color='#1f77b4')
    ax2.plot(dtensor_df['size_mb'], dtensor_df['throughput_mbs'], 's-', 
             label='Custom DTensor', markersize=6, linewidth=2, color='#ff7f0e')
    ax2.set_xlabel('Tensor Size (MB)', fontsize=12)
    ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax2.set_title('Tensor Size vs Throughput', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(script_dir, 'benchmark_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also show
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Size (MB)':<12} {'PyTorch (ms)':<15} {'DTensor (ms)':<15} {'Speedup':<10}")
    print("-"*60)
    
    for i in range(len(pytorch_df)):
        py_time = pytorch_df.iloc[i]['time_ms']
        dt_time = dtensor_df.iloc[i]['time_ms']
        size = pytorch_df.iloc[i]['size_mb']
        speedup = py_time / dt_time
        print(f"{size:<12.0f} {py_time:<15.3f} {dt_time:<15.3f} {speedup:<10.2f}x")
    
    print("="*60)
    avg_speedup = (pytorch_df['time_ms'] / dtensor_df['time_ms']).mean()
    print(f"Average Speedup: {avg_speedup:.2f}x")

if __name__ == "__main__":
    main()
