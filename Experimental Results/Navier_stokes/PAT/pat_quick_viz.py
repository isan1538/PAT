#!/usr/bin/env python3
"""
Quick and simple script to load trained PAT model and visualize results.

Usage:
    python pat_quick_viz.py --checkpoint checkpoints/pat_physics_best.pt
"""

import os
import argparse
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pat_visualize_results import (
    load_data, load_checkpoint, predict_at_time, compute_metrics,
    plot_fields, plot_scatter_comparison, plot_profiles
)


def create_summary_report(checkpoint_path, data_path, t_index, device, out_dir):
    """Create a comprehensive summary report with all visualizations"""
    
    print("="*80)
    print("PAT MODEL EVALUATION REPORT")
    print("="*80)
    
    # Load data
    print(f"\n1. Loading data from {data_path}...")
    data = load_data(data_path, device)
    
    # Load model
    print(f"\n2. Loading model from {checkpoint_path}...")
    model, cfg, mode, checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Print model info
    print(f"\n3. Model Configuration:")
    print(f"   Mode: {mode}")
    print(f"   Layers: {cfg.n_layer}")
    print(f"   Heads: {cfg.n_head}")
    print(f"   Embedding dim: {cfg.n_embd}")
    print(f"   Dropout: {cfg.dropout}")
    print(f"   Physics bias (alpha): {cfg.alpha}")
    
    if 'step' in checkpoint:
        print(f"   Training steps: {checkpoint['step']}")
    if 'best_mse' in checkpoint:
        print(f"   Best MSE: {checkpoint['best_mse']:.6e}")
    
    # Make predictions
    print(f"\n4. Generating predictions at t_index={t_index}...")
    results = predict_at_time(model, data, t_index, device, mode)
    
    # Compute metrics
    print("\n5. Computing evaluation metrics...")
    metrics = compute_metrics(results)
    
    # Print metrics
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print(f"Time: t={results['t']:.3f}")
    print(f"\nMean Squared Error (MSE):")
    print(f"  u-velocity: {metrics['mse_u']:.6e}")
    print(f"  v-velocity: {metrics['mse_v']:.6e}")
    print(f"  pressure:   {metrics['mse_p']:.6e}")
    print(f"  Average:    {metrics['mse_total']:.6e}")
    
    print(f"\nRelative L2 Error:")
    print(f"  u-velocity: {metrics['rel_l2_u']:.6f}")
    print(f"  v-velocity: {metrics['rel_l2_v']:.6f}")
    print(f"  pressure:   {metrics['rel_l2_p']:.6f}")
    print(f"  Average:    {metrics['rel_l2_avg']:.6f}")
    
    print(f"\nPeak Signal-to-Noise Ratio (PSNR):")
    print(f"  u-velocity: {metrics['psnr_u']:.2f} dB")
    print(f"  v-velocity: {metrics['psnr_v']:.2f} dB")
    print(f"  pressure:   {metrics['psnr_p']:.2f} dB")
    print(f"  Average:    {metrics['psnr_avg']:.2f} dB")
    print("="*80 + "\n")
    
    # Create visualizations
    print("6. Creating visualizations...")
    
    # Field plots
    save_path = os.path.join(out_dir, f'fields_{mode}_t{t_index}.png')
    plot_fields(results, metrics, mode, save_path)
    
    # Scatter comparison
    save_path = os.path.join(out_dir, f'scatter_{mode}_t{t_index}.png')
    plot_scatter_comparison(results, save_path)
    
    # Profile plots
    save_path = os.path.join(out_dir, f'profiles_{mode}_t{t_index}.png')
    plot_profiles(results, save_path)
    
    # Create a summary figure combining key information
    create_summary_figure(results, metrics, mode, checkpoint, out_dir, t_index)
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"Output directory: {out_dir}")
    print(f"\nGenerated files:")
    print(f"  - fields_{mode}_t{t_index}.png (detailed field visualizations)")
    print(f"  - scatter_{mode}_t{t_index}.png (prediction vs truth scatter)")
    print(f"  - profiles_{mode}_t{t_index}.png (1D profile comparisons)")
    print(f"  - summary_{mode}_t{t_index}.png (combined summary figure)")
    print("="*80 + "\n")


def create_summary_figure(results, metrics, mode, checkpoint, out_dir, t_index):
    """Create a single comprehensive summary figure"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    x = results['x']
    y = results['y']
    
    # Detect grid
    xu = np.unique(np.round(x, 10))
    yu = np.unique(np.round(y, 10))
    is_grid = len(x) == len(xu) * len(yu)
    
    if is_grid:
        nx, ny = len(xu), len(yu)
        x_to_i = {val: i for i, val in enumerate(xu)}
        y_to_j = {val: j for j, val in enumerate(yu)}
        
        def to_grid(values):
            grid = np.full((ny, nx), np.nan)
            for xi, yi, val in zip(np.round(x, 10), np.round(y, 10), values):
                grid[y_to_j[yi], x_to_i[xi]] = val
            return grid
        
        extent = [xu.min(), xu.max(), yu.min(), yu.max()]
    
    # Row 1: Ground truth fields
    for i, var in enumerate(['u', 'v', 'p']):
        ax = fig.add_subplot(gs[0, i])
        true_val = results[f'{var}_true']
        
        if is_grid:
            grid = to_grid(true_val)
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent,
                          cmap='RdBu_r', interpolation='bilinear')
        else:
            im = ax.scatter(x, y, c=true_val, s=2, cmap='RdBu_r')
            ax.set_aspect('equal')
        
        ax.set_title(f'{var} (Ground Truth)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: Predictions
    for i, var in enumerate(['u', 'v', 'p']):
        ax = fig.add_subplot(gs[1, i])
        pred_val = results[f'{var}_pred']
        
        if is_grid:
            grid = to_grid(pred_val)
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent,
                          cmap='RdBu_r', interpolation='bilinear')
        else:
            im = ax.scatter(x, y, c=pred_val, s=2, cmap='RdBu_r')
            ax.set_aspect('equal')
        
        ax.set_title(f'{var} (Prediction)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 3: Errors
    for i, var in enumerate(['u', 'v', 'p']):
        ax = fig.add_subplot(gs[2, i])
        true_val = results[f'{var}_true']
        pred_val = results[f'{var}_pred']
        err_val = np.abs(pred_val - true_val)
        
        if is_grid:
            grid = to_grid(err_val)
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent,
                          cmap='hot', interpolation='bilinear')
        else:
            im = ax.scatter(x, y, c=err_val, s=2, cmap='hot')
            ax.set_aspect('equal')
        
        ax.set_title(f'{var} (Absolute Error)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Summary info panel
    ax = fig.add_subplot(gs[:, 3])
    ax.axis('off')
    
    info_text = f"""
MODEL INFORMATION
{'='*40}
Mode: {mode}
Time: t = {results['t']:.3f}
Training step: {checkpoint.get('step', 'N/A')}

METRICS SUMMARY
{'='*40}
Relative L2 Error:
  u: {metrics['rel_l2_u']:.6f}
  v: {metrics['rel_l2_v']:.6f}
  p: {metrics['rel_l2_p']:.6f}
  avg: {metrics['rel_l2_avg']:.6f}

Mean Squared Error:
  u: {metrics['mse_u']:.3e}
  v: {metrics['mse_v']:.3e}
  p: {metrics['mse_p']:.3e}
  avg: {metrics['mse_total']:.3e}

PSNR (dB):
  u: {metrics['psnr_u']:.2f}
  v: {metrics['psnr_v']:.2f}
  p: {metrics['psnr_p']:.2f}
  avg: {metrics['psnr_avg']:.2f}

ERROR STATISTICS
{'='*40}
u-velocity:
  Mean: {np.mean(results['u_pred'] - results['u_true']):.3e}
  Std:  {np.std(results['u_pred'] - results['u_true']):.3e}
  Max:  {np.max(np.abs(results['u_pred'] - results['u_true'])):.3e}

v-velocity:
  Mean: {np.mean(results['v_pred'] - results['v_true']):.3e}
  Std:  {np.std(results['v_pred'] - results['v_true']):.3e}
  Max:  {np.max(np.abs(results['v_pred'] - results['v_true'])):.3e}

pressure:
  Mean: {np.mean(results['p_pred'] - results['p_true']):.3e}
  Std:  {np.std(results['p_pred'] - results['p_true']):.3e}
  Max:  {np.max(np.abs(results['p_pred'] - results['p_true'])):.3e}
    """
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f'PAT Model Evaluation Summary: {mode.upper()} Mode', 
                fontsize=16, fontweight='bold')
    
    save_path = os.path.join(out_dir, f'summary_{mode}_t{t_index}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Quick visualization of PAT model results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pat_quick_viz.py --checkpoint checkpoints/pat_physics_best.pt
  python pat_quick_viz.py --checkpoint checkpoints/pat_pure_best.pt --t_index 50
  python pat_quick_viz.py --checkpoint checkpoints/pat_physics_best.pt --out_dir my_results
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat',
                       help='Path to data file (default: ./cylinder_nektar_wake.mat)')
    parser.add_argument('--t_index', type=int, default=100,
                       help='Time index for evaluation (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Output directory (default: same as checkpoint dir)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.checkpoint)
        if not args.out_dir:
            args.out_dir = 'visualizations'
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create report
    create_summary_report(args.checkpoint, args.data_path, args.t_index, 
                         device, args.out_dir)


if __name__ == '__main__':
    main()
