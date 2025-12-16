#!/usr/bin/env python3
"""
Baseline Comparison Runner
==========================

Runs all baseline methods (PINN, SIREN, FNO, PI-DeepONet, PAT) and
compares results on sparse reconstruction task.

Usage:
    python run_comparison.py --M 200 --steps 5000

Author: Comparison suite for PAT research
"""

import os
import sys
import argparse
import time
import json
import subprocess
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# All methods to compare
METHODS = {
    'pinn': {
        'script': 'pinn_sparse.py',
        'name': 'PINN',
        'color': '#1f77b4',
        'description': 'Physics-Informed Neural Network'
    },
    'siren': {
        'script': 'siren_sparse.py',
        'name': 'SIREN',
        'color': '#ff7f0e',
        'description': 'Sinusoidal Representation Network'
    },
    'fno': {
        'script': 'fno_sparse.py',
        'name': 'FNO',
        'color': '#2ca02c',
        'description': 'Fourier Neural Operator'
    },
    'deeponet': {
        'script': 'deeponet_sparse.py',
        'name': 'PI-DeepONet',
        'color': '#d62728',
        'description': 'Physics-Informed DeepONet'
    },
    'pat': {
        'script': 'pat_training_1208.py',
        'name': 'PAT',
        'color': '#9467bd',
        'description': 'Physics-Aware Transformer'
    }
}


def run_method(method_key, args):
    """
    Run a single method and return results.
    """
    method = METHODS[method_key]
    print(f"\n{'='*80}")
    print(f"Running {method['name']}: {method['description']}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        'python', method['script'],
        '--M', str(args.M),
        '--steps', str(args.steps),
        '--modes'] + [str(m) for m in args.modes] + [
        '--nu', str(args.nu),
        '--device', args.device,
        '--save_path', f"checkpoints/{method_key}_M{args.M}.pt"
    ]
    
    # Add method-specific args
    if method_key == 'pat':
        cmd.extend(['--warmup_steps', str(args.steps // 10)])
    
    # Run
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time
        
        # Parse output for final metrics
        output = result.stdout
        print(output)
        
        # Extract metrics from output
        mse = None
        mae = None
        for line in output.split('\n'):
            if 'Final MSE:' in line:
                mse = float(line.split(':')[1].strip())
            elif 'Final MAE:' in line:
                mae = float(line.split(':')[1].strip())
        
        return {
            'method': method['name'],
            'mse': mse,
            'mae': mae,
            'time': elapsed,
            'success': True,
            'error': None
        }
    
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\nERROR running {method['name']}:")
        print(e.stderr)
        
        return {
            'method': method['name'],
            'mse': None,
            'mae': None,
            'time': elapsed,
            'success': False,
            'error': str(e)
        }


def plot_comparison(results, save_path='comparison_results.png'):
    """
    Create comparison plots.
    """
    # Filter successful results
    success_results = [r for r in results if r['success']]
    
    if len(success_results) == 0:
        print("No successful results to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [r['method'] for r in success_results]
    mses = [r['mse'] for r in success_results]
    maes = [r['mae'] for r in success_results]
    times = [r['time'] / 60 for r in success_results]  # Convert to minutes
    
    colors = [METHODS[k]['color'] for k in METHODS.keys() 
              if METHODS[k]['name'] in methods]
    
    # MSE comparison
    axes[0].bar(methods, mses, color=colors)
    axes[0].set_ylabel('MSE (log scale)')
    axes[0].set_title('Mean Squared Error')
    axes[0].set_yscale('log')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE comparison
    axes[1].bar(methods, maes, color=colors)
    axes[1].set_ylabel('MAE (log scale)')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Training time
    axes[2].bar(methods, times, color=colors)
    axes[2].set_ylabel('Time (minutes)')
    axes[2].set_title('Training Time')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {save_path}")


def create_table(results, save_path='comparison_table.csv'):
    """
    Create comparison table.
    """
    df = pd.DataFrame(results)
    df = df[['method', 'mse', 'mae', 'time', 'success']]
    df['time_min'] = df['time'] / 60
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nComparison table saved to {save_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Method':<15} {'MSE':<12} {'MAE':<12} {'Time (min)':<12} {'Status'}")
    print("-"*80)
    
    for r in results:
        status = '✓' if r['success'] else '✗'
        mse_str = f"{r['mse']:.4e}" if r['mse'] is not None else 'N/A'
        mae_str = f"{r['mae']:.4e}" if r['mae'] is not None else 'N/A'
        time_str = f"{r['time']/60:.1f}" if r['time'] is not None else 'N/A'
        
        print(f"{r['method']:<15} {mse_str:<12} {mae_str:<12} {time_str:<12} {status}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison")
    
    # Problem setup
    parser.add_argument("--M", type=int, default=200, 
                       help="Number of sparse observations")
    parser.add_argument("--steps", type=int, default=5000,
                       help="Training steps")
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2, 3],
                       help="Fourier modes")
    parser.add_argument("--nu", type=float, default=0.1,
                       help="Thermal diffusivity")
    
    # Methods to run
    parser.add_argument("--methods", nargs="+", 
                       choices=list(METHODS.keys()) + ['all'],
                       default=['all'],
                       help="Methods to run (default: all)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                       help="Directory for results")
    
    args = parser.parse_args()
    
    # Determine which methods to run
    if 'all' in args.methods:
        methods_to_run = list(METHODS.keys())
    else:
        methods_to_run = args.methods
    
    print(f"\n{'='*80}")
    print(f"BASELINE COMPARISON - Sparse Reconstruction")
    print(f"{'='*80}")
    print(f"Sparse points (M): {args.M}")
    print(f"Training steps: {args.steps}")
    print(f"Methods: {', '.join([METHODS[m]['name'] for m in methods_to_run])}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run each method
    results = []
    for method_key in methods_to_run:
        result = run_method(method_key, args)
        results.append(result)
        
        # Save intermediate results
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Create comparison plots
    plot_comparison(results, 
                   os.path.join(args.output_dir, 'comparison_plot.png'))
    
    # Create table
    create_table(results,
                os.path.join(args.output_dir, 'comparison_table.csv'))
    
    # Save configuration
    config = {
        'M': args.M,
        'steps': args.steps,
        'modes': args.modes,
        'nu': args.nu,
        'timestamp': datetime.now().isoformat(),
        'methods': methods_to_run
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nAll results saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()