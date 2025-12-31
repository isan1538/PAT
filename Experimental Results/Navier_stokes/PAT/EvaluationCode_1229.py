#!/usr/bin/env python3
"""
Load trained PAT model and evaluate with comprehensive metrics


python evaluate_model.py --checkpoint checkpoints/pat_physics_best.pt
"""

import os
import argparse
import torch
import numpy as np

from pat_model_1208 import PATConfig
from your_training_file import (
    PATModelNS, 
    CylinderWakeData,
    compute_additional_metrics,
    to_device
)


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: torch device
    
    Returns:
        model: Loaded PATModelNS model
        checkpoint: Full checkpoint dict with metadata
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    cfg_dict = checkpoint['cfg']  # or checkpoint['config'] depending on your save format
    cfg = PATConfig(**cfg_dict)
    
    # Get mode
    mode = checkpoint.get('mode', checkpoint['args'].get('mode', 'physics'))
    
    # Determine output dimension
    out_dim = 2 if mode == 'streamfunction' else 3
    
    # Create model
    model = PATModelNS(cfg, out_dim=out_dim, mode=mode)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])  # or checkpoint['model_state_dict']
    model = model.to(device)
    model.eval()
    
    print(f"  Mode: {mode}")
    print(f"  Training step: {checkpoint.get('step', 'N/A')}")
    print(f"  Best MSE: {checkpoint.get('eval_mse', 'N/A')}")
    
    return model, checkpoint


def evaluate_loaded_model(checkpoint_path, data_path, eval_t_index=100, Re=100.0, device='cuda'):
    """
    Complete evaluation pipeline
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model, checkpoint = load_model(checkpoint_path, device)
    
    # Get mode from checkpoint
    if 'mode' in checkpoint:
        mode = checkpoint['mode']
    elif 'args' in checkpoint:
        mode = checkpoint['args'].get('mode', 'physics')
    else:
        mode = 'physics'  # default
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    data = CylinderWakeData(data_path, seed=0)
    
    # Calculate nu
    nu = 1.0 / Re
    
    # Compute metrics
    print("\nComputing comprehensive metrics...")
    metrics = compute_additional_metrics(model, data, device, eval_t_index, nu, mode=mode)
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION METRICS")
    print(f"{'='*80}")
    print(f"Evaluation at t_index: {eval_t_index}")
    print(f"Reynolds number: {Re}")
    print(f"Mode: {mode}")
    print(f"\nRelative L2 Errors:")
    print(f"  u: {metrics['rel_l2_u']:.6f}")
    print(f"  v: {metrics['rel_l2_v']:.6f}")
    print(f"  p: {metrics['rel_l2_p']:.6f}")
    print(f"  avg: {metrics['rel_l2_avg']:.6f}")
    print(f"\nPSNR:")
    print(f"  u: {metrics['psnr_u']:.2f} dB")
    print(f"  v: {metrics['psnr_v']:.2f} dB")
    print(f"  p: {metrics['psnr_p']:.2f} dB")
    print(f"  avg: {metrics['psnr_avg']:.2f} dB")
    print(f"\nPDE Residual: {metrics['pde_residual']:.6e}")
    print(f"{'='*80}\n")
    
    # Optionally save metrics back to checkpoint
    save_metrics = input("Save metrics to checkpoint? (y/n): ")
    if save_metrics.lower() == 'y':
        checkpoint['metrics'] = metrics
        torch.save(checkpoint, checkpoint_path)
        print(f"Metrics saved to: {checkpoint_path}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PAT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat',
                       help='Path to data file')
    parser.add_argument('--eval_t_index', type=int, default=100,
                       help='Time index for evaluation')
    parser.add_argument('--Re', type=float, default=100.0,
                       help='Reynolds number')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluate_loaded_model(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        eval_t_index=args.eval_t_index,
        Re=args.Re,
        device=args.device
    )


if __name__ == '__main__':
    main()