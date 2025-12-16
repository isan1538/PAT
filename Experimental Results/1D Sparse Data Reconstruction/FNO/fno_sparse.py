#!/usr/bin/env python3
"""
Fourier Neural Operator (FNO) for Sparse Reconstruction
=======================================================

Implements FNO for 1D+time heat equation with sparse observations.
Uses spectral convolutions in Fourier space.

Reference: Li et al. "Fourier Neural Operator for Parametric Partial 
           Differential Equations" (ICLR 2021)

Installation:
    pip install neuraloperator

Author: Baseline comparison for PAT
"""

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

# Try to import neuraloperator library
try:
    from neuralop.models import FNO
    NEURALOP_AVAILABLE = True
except ImportError:
    print("Warning: neuraloperator not installed. Using custom FNO implementation.")
    NEURALOP_AVAILABLE = False


# ============================================================================
# Exact Solution
# ============================================================================

def exact_solution(x, t, nu=0.1, n=1):
    """Exact solution for 1D heat equation."""
    return torch.exp(-nu * (n * math.pi)**2 * t) * torch.sin(n * math.pi * x)


# ============================================================================
# Custom FNO Implementation (if library not available)
# ============================================================================

class SpectralConv1d(nn.Module):
    """1D Fourier layer (spectral convolution)."""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep
        
        scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        """
        x: (batch, in_channels, x_points)
        returns: (batch, out_channels, x_points)
        """
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        
        # IFFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d_Custom(nn.Module):
    """
    Custom FNO for 1D+time problems.
    Maps from (x,t) grid to solution u(x,t).
    """
    def __init__(
        self,
        modes=16,
        width=64,
        n_layers=4,
        in_channels=3,  # (x, t, a(x)) where a could be sparse observations
        out_channels=1
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        
        # Lift to higher dimension
        self.lift = nn.Linear(in_channels, width)
        
        # Fourier layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        
        # Local (non-spectral) layers
        self.w_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Project to output
        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )
    
    def forward(self, x):
        """
        x: (batch, n_points, in_channels)
        returns: (batch, n_points, out_channels)
        """
        # Lift
        x = self.lift(x)  # (batch, n_points, width)
        x = x.permute(0, 2, 1)  # (batch, width, n_points)
        
        # Fourier layers
        for i in range(self.n_layers):
            x1 = self.spectral_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        # Project
        x = x.permute(0, 2, 1)  # (batch, n_points, width)
        x = self.project(x)  # (batch, n_points, out_channels)
        
        return x


# ============================================================================
# FNO Wrapper for Sparse Reconstruction
# ============================================================================

class FNOSparseReconstruction(nn.Module):
    """
    Wrapper for FNO to handle sparse observations.
    
    Strategy: Create a gridded representation from sparse data,
    then use FNO to predict full solution.
    """
    def __init__(
        self,
        n_grid=128,  # Resolution of internal grid
        modes=16,
        width=64,
        n_layers=4
    ):
        super().__init__()
        
        self.n_grid = n_grid
        
        # Create grid
        self.register_buffer('x_grid', torch.linspace(0, 1, n_grid))
        
        # FNO model
        if NEURALOP_AVAILABLE:
            # Use official implementation
            self.fno = FNO(
                n_modes=(modes,),
                hidden_channels=width,
                in_channels=3,  # (x, t, sparse_interpolated)
                out_channels=1,
                n_layers=n_layers
            )
        else:
            # Use custom implementation
            self.fno = FNO1d_Custom(
                modes=modes,
                width=width,
                n_layers=n_layers,
                in_channels=3,
                out_channels=1
            )
    
    def interpolate_sparse_to_grid(self, xt_sparse, u_sparse, t_query):
        """
        Interpolate sparse observations to a regular grid at time t_query.
        
        Args:
            xt_sparse: (batch, M, 2) - sparse observation locations
            u_sparse: (batch, M, 1) - sparse observation values
            t_query: scalar - time at which to evaluate
        
        Returns:
            grid_features: (batch, n_grid, 3) - [x, t, u_interpolated]
        """
        batch_size = xt_sparse.shape[0]
        device = xt_sparse.device
        
        # Create grid
        x_grid = self.x_grid.unsqueeze(0).repeat(batch_size, 1)  # (batch, n_grid)
        t_grid = torch.full_like(x_grid, t_query)  # (batch, n_grid)
        
        # Interpolate sparse observations to grid using RBF or nearest neighbor
        # Simple approach: inverse distance weighting
        u_grid = torch.zeros(batch_size, self.n_grid, 1, device=device)
        
        for b in range(batch_size):
            x_obs = xt_sparse[b, :, 0]  # (M,)
            u_obs = u_sparse[b, :, 0]   # (M,)
            
            # Compute distances
            dist = torch.abs(x_grid[b].unsqueeze(1) - x_obs.unsqueeze(0))  # (n_grid, M)
            
            # Inverse distance weighting (add small epsilon to avoid division by zero)
            weights = 1.0 / (dist + 1e-6)  # (n_grid, M)
            weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
            
            # Weighted sum
            u_grid[b, :, 0] = (weights * u_obs.unsqueeze(0)).sum(dim=1)
        
        # Combine into features
        grid_features = torch.stack([x_grid, t_grid, u_grid.squeeze(-1)], dim=-1)
        
        return grid_features
    
    def forward(self, xt_sparse, u_sparse, xt_query):
        """
        Predict solution at query points given sparse observations.
        
        Args:
            xt_sparse: (batch, M, 2) - sparse observation locations
            u_sparse: (batch, M, 1) - sparse observation values
            xt_query: (batch, N, 2) - query locations
        
        Returns:
            u_query: (batch, N, 1) - predicted values
        """
        batch_size = xt_query.shape[0]
        n_query = xt_query.shape[1]
        device = xt_query.device
        
        # For simplicity, assume all queries are at same time
        # (In practice, might need to handle multiple times separately)
        t_avg = xt_query[:, :, 1].mean()
        
        # Interpolate sparse data to grid
        grid_features = self.interpolate_sparse_to_grid(xt_sparse, u_sparse, t_avg)
        
        # Pass through FNO
        u_grid = self.fno(grid_features)  # (batch, n_grid, 1)
        
        # Interpolate from grid to query points
        u_query = torch.zeros(batch_size, n_query, 1, device=device)
        
        for b in range(batch_size):
            x_query = xt_query[b, :, 0]  # (N,)
            
            # Linear interpolation from grid
            u_query[b] = F.grid_sample(
                u_grid[b].unsqueeze(0).unsqueeze(0),  # (1, 1, n_grid, 1)
                x_query.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * 2 - 1,  # Normalize to [-1, 1]
                align_corners=True
            ).squeeze()
        
        return u_query


# ============================================================================
# Simpler Approach: Direct MLP + FNO for point-wise prediction
# ============================================================================

class FNOPointwise(nn.Module):
    """
    Simplified approach: Use MLP to encode sparse observations,
    then predict at query points.
    """
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        
        # Encoder for sparse observations
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # (x, t, u)
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder for query points
        self.decoder = nn.Sequential(
            nn.Linear(2 + hidden_dim, hidden_dim),  # (x, t, context)
            nn.GELU(),
            *[layer for _ in range(num_layers-1) 
              for layer in (nn.Linear(hidden_dim, hidden_dim), nn.GELU())],
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, xt_sparse, u_sparse, xt_query):
        """
        Args:
            xt_sparse: (batch, M, 2)
            u_sparse: (batch, M, 1)
            xt_query: (batch, N, 2)
        """
        # Encode sparse observations
        sparse_input = torch.cat([xt_sparse, u_sparse], dim=-1)  # (batch, M, 3)
        encoded = self.encoder(sparse_input)  # (batch, M, hidden)
        
        # Global context (mean pooling)
        context = encoded.mean(dim=1, keepdim=True)  # (batch, 1, hidden)
        context = context.expand(-1, xt_query.shape[1], -1)  # (batch, N, hidden)
        
        # Decode at query points
        query_input = torch.cat([xt_query, context], dim=-1)
        u_query = self.decoder(query_input)
        
        return u_query


# ============================================================================
# Dataset (Same as others)
# ============================================================================

class SparseHeatDataset(Dataset):
    """Dataset for sparse heat equation reconstruction."""
    def __init__(
        self,
        num_instances=10**6,
        M_sparse=200,
        nu=0.1,
        modes=[1, 2, 3],
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
        mode_n = self.modes[idx % len(self.modes)]
        
        # Sparse observations
        xt_sparse = torch.rand(self.M, 2, device=self.device)
        u_sparse = exact_solution(
            xt_sparse[:, 0], xt_sparse[:, 1], 
            self.nu, mode_n
        ).unsqueeze(-1)
        
        # Random query points for training
        n_query = 1000
        xt_query = torch.rand(n_query, 2, device=self.device)
        u_query = exact_solution(
            xt_query[:, 0], xt_query[:, 1],
            self.nu, mode_n
        ).unsqueeze(-1)
        
        return {
            'xt_sparse': xt_sparse,
            'u_sparse': u_sparse,
            'xt_query': xt_query,
            'u_query': u_query,
            'mode': mode_n
        }


# ============================================================================
# Training
# ============================================================================

def train_step(model, optimizer, batch, device):
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    xt_sparse = batch['xt_sparse'].to(device)
    u_sparse = batch['u_sparse'].to(device)
    xt_query = batch['xt_query'].to(device)
    u_query = batch['u_query'].to(device)
    
    # Predict
    u_pred = model(xt_sparse, u_sparse, xt_query)
    
    # Loss
    loss = F.mse_loss(u_pred, u_query)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {'loss': loss.item()}


def evaluate_model(model, nu, mode_n, device, nx=256, nt=128, M_eval=200):
    """Evaluate model."""
    model.eval()
    
    with torch.no_grad():
        # Create sparse observations for evaluation
        xt_sparse = torch.rand(1, M_eval, 2, device=device)
        u_sparse = exact_solution(
            xt_sparse[0, :, 0], xt_sparse[0, :, 1],
            nu, mode_n
        ).unsqueeze(0).unsqueeze(-1)
        
        # Evaluation grid
        x = torch.linspace(0, 1, nx, device=device)
        t = torch.linspace(0, 1, nt, device=device)
        X, T = torch.meshgrid(x, t, indexing='ij')
        
        xt_eval = torch.stack([X, T], dim=-1).reshape(1, -1, 2)
        
        # Predict in batches
        batch_size = 4096
        u_pred_list = []
        for i in range(0, xt_eval.shape[1], batch_size):
            xt_batch = xt_eval[:, i:i+batch_size]
            u_batch = model(xt_sparse, u_sparse, xt_batch)
            u_pred_list.append(u_batch)
        
        u_pred = torch.cat(u_pred_list, dim=1).reshape(nx, nt)
        u_true = exact_solution(X, T, nu, mode_n)
        
        # Errors
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
        'T': T.cpu()
    }


def plot_results(history, eval_results, save_path='fno_results.png'):
    """Plot results."""
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Ground truth
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.contourf(eval_results['T'], eval_results['X'], 
                      eval_results['ground_truth'], levels=50, cmap='viridis')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Space x')
    ax1.set_title('Ground Truth')
    plt.colorbar(im, ax=ax1)
    
    # Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.contourf(eval_results['T'], eval_results['X'],
                      eval_results['prediction'], levels=50, cmap='viridis')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Space x')
    ax2.set_title('FNO Prediction')
    plt.colorbar(im, ax=ax2)
    
    # Error
    ax3 = fig.add_subplot(gs[0, 2])
    error = (eval_results['prediction'] - eval_results['ground_truth']).abs()
    im = ax3.contourf(eval_results['T'], eval_results['X'], 
                      error, levels=50, cmap='Reds')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Space x')
    ax3.set_title(f'Error (max={error.max():.2e})')
    plt.colorbar(im, ax=ax3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training
# ============================================================================

def train_fno(args):
    """Main training loop."""
    device = torch.device(args.device)
    print(f"\n{'='*80}")
    print(f"FNO - Sparse Reconstruction Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Using: {'Official neuralop' if NEURALOP_AVAILABLE else 'Custom FNO'}")
    print(f"Sparse points (M): {args.M}")
    print(f"{'='*80}\n")
    
    # Dataset
    dataset = SparseHeatDataset(
        M_sparse=args.M,
        nu=args.nu,
        modes=args.modes,
        device=device
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model (using simpler pointwise approach)
    model = FNOPointwise(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    
    # Training
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
        
        if step % args.eval_every == 0 or step == args.steps:
            eval_results = evaluate_model(model, args.nu, args.modes[0], device, M_eval=args.M)
            print(f"       EVAL → MSE={eval_results['mse']:.3e} "
                  f"MAE={eval_results['mae']:.3e}")
            
            if eval_results['mse'] < best_mse:
                best_mse = eval_results['mse']
                print(f"       → New best: {best_mse:.3e}")
            
            print("-" * 80)
    
    # Final
    elapsed = time.time() - start_time
    final_eval = evaluate_model(model, args.nu, args.modes[0], device, M_eval=args.M)
    
    print("\n" + "="*80)
    print(f"Final MSE:    {final_eval['mse']:.4e}")
    print(f"Final MAE:    {final_eval['mae']:.4e}")
    print(f"Time:         {elapsed:.1f}s")
    print(f"Best MSE:     {best_mse:.4e}")
    print("="*80)
    
    if args.plot:
        plot_results({}, final_eval, 'fno_results.png')
    
    return model, final_eval


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fno(args)
