#!/usr/bin/env python3
'''
FNO Baseline for 2D Heat Equation - Sparse Reconstruction

2D Heat Equation: ∂u/∂t = ν(∂²u/∂x² + ∂²u/∂y²)
Domain: (x,y,t) ∈ [0,1]² × [0,1]

Example usage:
python 1212_fno_2d.py \
    --M 100 \
    --nu 0.1 \
    --train_modes "1,1" "1,2" \
    --test_modes "2,2" "2,3" \
    --hidden_dim 256 \
    --num_layers 6 \
    --steps 30000 \
    --batch_size 4 \
    --lr 1e-3 \
    --print_every 100 \
    --eval_every 500 \
    --plot
'''

import os
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def exact_solution_2d(x, y, t, nu=0.1, n=1, m=1):
    """
    Analytical solution for 2D heat equation with separable variables.
    
    u(x,y,t) = exp(-ν*π²*(n²+m²)*t) * sin(n*π*x) * sin(m*π*y)
    """
    factor = nu * (math.pi ** 2) * (n**2 + m**2)
    return torch.exp(-factor * t) * torch.sin(n * math.pi * x) * torch.sin(m * math.pi * y)


class FNOPointwise2D(nn.Module):
    """
    Pointwise neural operator for 2D heat equation.
    Maps sparse observations to arbitrary query points.
    """
    
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        
        # Encoder: processes sparse observations (x, y, t, u)
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (x, y, t, u) -> hidden
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder: processes query points with context
        decoder_layers = [
            nn.Linear(3 + hidden_dim, hidden_dim),  # (x, y, t) + context
            nn.GELU()
        ]
        for _ in range(num_layers - 1):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ])
        decoder_layers.append(nn.Linear(hidden_dim, 1))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, xyt_sparse, u_sparse, xyt_query):
        """
        Args:
            xyt_sparse: [B, M, 3] sparse observation positions
            u_sparse: [B, M, 1] sparse observation values
            xyt_query: [B, N, 3] query positions
        
        Returns:
            u_query: [B, N, 1] predictions at query positions
        """
        # Encode sparse observations
        sparse_input = torch.cat([xyt_sparse, u_sparse], dim=-1)  # [B, M, 4]
        encoded = self.encoder(sparse_input)  # [B, M, hidden_dim]
        
        # Global context from sparse observations
        context = encoded.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        context = context.expand(-1, xyt_query.shape[1], -1)  # [B, N, hidden_dim]
        
        # Decode query points with context
        query_input = torch.cat([xyt_query, context], dim=-1)  # [B, N, 3+hidden_dim]
        u_query = self.decoder(query_input)  # [B, N, 1]
        
        return u_query


class SparseHeat2DDataset(Dataset):
    """
    Dataset for 2D heat equation sparse reconstruction.
    Generates random sparse observations and query points on-the-fly.
    """
    
    def __init__(
        self,
        num_instances=10**6,
        M_sparse=100,
        nu=0.1,
        modes=[(1, 1), (1, 2)],  # List of (n, m) tuples
        device='cpu'
    ):
        super().__init__()
        self.num_instances = num_instances
        self.M = M_sparse
        self.nu = nu
        self.modes = modes
        self.device = torch.device(device)
    
    def __len__(self):
        return self.num_instances
    
    def __getitem__(self, idx):
        # Select mode
        n, m = self.modes[idx % len(self.modes)]
        
        # Generate sparse observations
        xyt_sparse = torch.rand(self.M, 3, device=self.device)
        u_sparse = exact_solution_2d(
            xyt_sparse[:, 0], xyt_sparse[:, 1], xyt_sparse[:, 2],
            self.nu, n, m
        ).unsqueeze(-1)
        
        # Generate query points
        n_query = 1000
        xyt_query = torch.rand(n_query, 3, device=self.device)
        u_query = exact_solution_2d(
            xyt_query[:, 0], xyt_query[:, 1], xyt_query[:, 2],
            self.nu, n, m
        ).unsqueeze(-1)
        
        return {
            'xyt_sparse': xyt_sparse,
            'u_sparse': u_sparse,
            'xyt_query': xyt_query,
            'u_query': u_query,
            'mode': (n, m)
        }


def train_step(model, optimizer, batch, device):
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    xyt_sparse = batch['xyt_sparse'].to(device)
    u_sparse = batch['u_sparse'].to(device)
    xyt_query = batch['xyt_query'].to(device)
    u_query = batch['u_query'].to(device)
    
    u_pred = model(xyt_sparse, u_sparse, xyt_query)
    
    loss = F.mse_loss(u_pred, u_query)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {'loss': loss.item()}


def evaluate_model_2d(model, nu, mode, device, nx=64, ny=64, M_eval=100):
    """
    Evaluate model on 2D heat equation at fixed time slice.
    
    Args:
        model: FNO model
        nu: diffusivity
        mode: tuple (n, m) for Fourier mode
        device: torch device
        nx, ny: evaluation grid resolution
        M_eval: number of sparse observations
    """
    model.eval()
    n, m = mode
    
    with torch.no_grad():
        # Generate sparse observations
        xyt_sparse = torch.rand(1, M_eval, 3, device=device)
        u_sparse = exact_solution_2d(
            xyt_sparse[0, :, 0], xyt_sparse[0, :, 1], xyt_sparse[0, :, 2],
            nu, n, m
        ).unsqueeze(0).unsqueeze(-1)
        
        # Evaluation grid at fixed time
        t_eval = 0.5
        x = torch.linspace(0, 1, nx, device=device)
        y = torch.linspace(0, 1, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        T = torch.full_like(X, t_eval)
        
        xyt_eval = torch.stack([X, Y, T], dim=-1).reshape(1, -1, 3)
        
        # Predict in batches to avoid memory issues
        batch_size = 4096
        u_pred_list = []
        for i in range(0, xyt_eval.shape[1], batch_size):
            xyt_batch = xyt_eval[:, i:i+batch_size]
            u_batch = model(xyt_sparse, u_sparse, xyt_batch)
            u_pred_list.append(u_batch)
        
        u_pred = torch.cat(u_pred_list, dim=1).reshape(nx, ny)
        u_true = exact_solution_2d(X, Y, T, nu, n, m)
        
        # Compute metrics
        mse = F.mse_loss(u_pred, u_true).item()
        mae = (u_pred - u_true).abs().mean().item()
        
        threshold = 1e-3
        mask = u_true.abs() > threshold
        if mask.sum() > 0:
            rel_error = ((u_pred[mask] - u_true[mask]).abs() / u_true[mask].abs()).mean().item()
        else:
            rel_error = float('nan')
        
        max_error = (u_pred - u_true).abs().max().item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rel_error': rel_error,
        'max_error': max_error,
        'prediction': u_pred.cpu(),
        'ground_truth': u_true.cpu(),
        'X': X.cpu(),
        'Y': Y.cpu(),
        't_eval': t_eval
    }


def plot_results_2d(history, eval_results, save_path='fno_2d_results.png', mode=None):
    """Plot training history and 2D solution comparison."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    if len(history['step']) > 0:
        ax1.semilogy(history['step'], history['loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
    
    # Train vs Test MSE
    ax2 = fig.add_subplot(gs[0, 1])
    if len(history['eval_mse']) > 0:
        ax2.semilogy(history['eval_step'], history['eval_mse'], 'b-o', label='Train MSE', markersize=4)
        if len(history['test_mse']) > 0:
            ax2.semilogy(history['test_step'], history['test_mse'], 'r-o', label='Test MSE', markersize=4)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('MSE')
        ax2.set_title('Train vs Test MSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Relative Error
    ax3 = fig.add_subplot(gs[0, 2])
    if len(history['eval_rel']) > 0:
        ax3.plot(history['eval_step'], [r*100 for r in history['eval_rel']], 
                 'b-o', label='Train RelErr', markersize=4)
        if len(history['test_rel']) > 0:
            ax3.plot(history['test_step'], [r*100 for r in history['test_rel']], 
                     'r-o', label='Test RelErr', markersize=4)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Relative Error (%)')
        ax3.set_title('Relative Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Ground truth
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(eval_results['ground_truth'].T, origin='lower', 
                    aspect='auto', cmap='viridis', extent=[0, 1, 0, 1])
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title(f'Ground Truth (t={eval_results["t_eval"]:.2f})')
    plt.colorbar(im, ax=ax4)
    
    # Prediction
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(eval_results['prediction'].T, origin='lower',
                    aspect='auto', cmap='viridis', extent=[0, 1, 0, 1])
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title(f'FNO Prediction (t={eval_results["t_eval"]:.2f})')
    plt.colorbar(im, ax=ax5)
    
    # Error
    ax6 = fig.add_subplot(gs[1, 2])
    error = (eval_results['prediction'] - eval_results['ground_truth']).abs()
    im = ax6.imshow(error.T, origin='lower', aspect='auto', cmap='Reds', extent=[0, 1, 0, 1])
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title(f'Absolute Error (max={error.max():.2e})')
    plt.colorbar(im, ax=ax6)
    
    if mode:
        fig.suptitle(f'FNO 2D - Mode ({mode[0]},{mode[1]})', fontsize=14, y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to {save_path}")


def train_fno_2d(args):
    """Main training loop for 2D FNO."""
    device = torch.device(args.device)
    
    # Parse modes from string format "n,m"
    def parse_mode(mode_str):
        return tuple(map(int, mode_str.split(",")))
    
    train_modes = [parse_mode(m) for m in args.train_modes]
    test_modes = [parse_mode(m) for m in args.test_modes]
    
    print(f"\n{'='*80}")
    print(f"FNO 2D - Sparse Reconstruction Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Sparse points (M): {args.M}")
    print(f"Training modes: {train_modes}")
    print(f"Test modes: {test_modes}")
    print(f"{'='*80}\n")
    
    # Dataset
    dataset = SparseHeat2DDataset(
        M_sparse=args.M,
        nu=args.nu,
        modes=train_modes,
        device=device
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = FNOPointwise2D(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    
    # Training history
    history = {
        'step': [], 'loss': [],
        'eval_step': [], 'eval_mse': [], 'eval_mae': [], 'eval_rel': [],
        'test_step': [], 'test_mse': [], 'test_mae': [], 'test_rel': []
    }
    
    print("Starting training...")
    print("-" * 80)
    
    best_mse = float('inf')
    data_iter = iter(loader)
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        stats = train_step(model, optimizer, batch, device)
        scheduler.step()
        
        if step % args.print_every == 0 or step == 1:
            print(f"[{step:05d}] loss={stats['loss']:.3e}")
            history['step'].append(step)
            history['loss'].append(stats['loss'])
        
        if step % args.eval_every == 0 or step == args.steps:
            # Eval on seen mode
            eval_seen = evaluate_model_2d(model, args.nu, train_modes[0], device, M_eval=args.M)
            print(f"       EVAL (seen mode {train_modes[0]}) → MSE={eval_seen['mse']:.3e} "
                  f"MAE={eval_seen['mae']:.3e} RelErr={eval_seen['rel_error']:.3%}")
            
            history['eval_step'].append(step)
            history['eval_mse'].append(eval_seen['mse'])
            history['eval_mae'].append(eval_seen['mae'])
            history['eval_rel'].append(eval_seen['rel_error'])
            
            # Test on unseen mode
            test_unseen = evaluate_model_2d(model, args.nu, test_modes[0], device, M_eval=args.M)
            print(f"       TEST (unseen mode {test_modes[0]}) → MSE={test_unseen['mse']:.3e} "
                  f"MAE={test_unseen['mae']:.3e} RelErr={test_unseen['rel_error']:.3%}")
            
            history['test_step'].append(step)
            history['test_mse'].append(test_unseen['mse'])
            history['test_mae'].append(test_unseen['mae'])
            history['test_rel'].append(test_unseen['rel_error'])
            
            if eval_seen['mse'] < best_mse:
                best_mse = eval_seen['mse']
            
            print("-" * 80)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    print("\n--- Interpolation (Seen Modes) ---")
    for mode in train_modes:
        eval_seen = evaluate_model_2d(model, args.nu, mode, device, nx=128, ny=128, M_eval=args.M)
        print(f"Mode {mode}: MSE={eval_seen['mse']:.4e}, MAE={eval_seen['mae']:.4e}, "
              f"RelErr={eval_seen['rel_error']:.4e}")
    
    print("\n--- Extrapolation (Unseen Modes) ---")
    for mode in test_modes:
        eval_unseen = evaluate_model_2d(model, args.nu, mode, device, nx=128, ny=128, M_eval=args.M)
        print(f"Mode {mode}: MSE={eval_unseen['mse']:.4e}, MAE={eval_unseen['mae']:.4e}, "
              f"RelErr={eval_unseen['rel_error']:.4e}")
    
    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"Best training MSE: {best_mse:.4e}")
    print("="*80)
    
    # Plot results
    if args.plot:
        final_eval = evaluate_model_2d(model, args.nu, train_modes[0], device, 
                                       nx=128, ny=128, M_eval=args.M)
        plot_results_2d(history, final_eval, 'fno_2d_seen_mode.png', mode=train_modes[0])
        
        unseen_eval = evaluate_model_2d(model, args.nu, test_modes[0], device,
                                        nx=128, ny=128, M_eval=args.M)
        plot_results_2d(history, unseen_eval, 'fno_2d_unseen_mode.png', mode=test_modes[0])
    
    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO for 2D heat equation")
    
    # Problem parameters
    parser.add_argument("--nu", type=float, default=0.1, help="Diffusivity")
    parser.add_argument("--train_modes", nargs="+", type=str, default=["1,1", "1,2"],
                        help="Training modes as 'n,m' strings")
    parser.add_argument("--test_modes", nargs="+", type=str, default=["2,2", "2,3"],
                        help="Test modes as 'n,m' strings")
    parser.add_argument("--M", type=int, default=100, help="Number of sparse observations")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of decoder layers")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Logging
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--plot", action="store_true", default=True)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fno_2d(args)
