#!/usr/bin/env python3
"""
Physics-Informed Neural Network (PINN) for Sparse Reconstruction
=================================================================

Implements vanilla PINN for 1D heat equation with sparse observations.
Compares directly with PAT on the same sparse reconstruction task.

Reference: Raissi et al. "Physics-informed neural networks: A deep learning 
           framework for solving forward and inverse problems involving 
           nonlinear partial differential equations" (2019)

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


# ============================================================================
# Exact Solution
# ============================================================================

def exact_solution(x, t, nu=0.1, n=1):
    """
    Exact solution for 1D heat equation:
    u(x,t) = exp(-nu*(n*pi)^2 * t) * sin(n*pi*x)
    """
    return torch.exp(-nu * (n * math.pi)**2 * t) * torch.sin(n * math.pi * x)


# ============================================================================
# PINN Model Architecture
# ============================================================================

class PINN(nn.Module):
    """
    Standard PINN architecture with tanh activation.
    
    Input: (x, t) coordinates
    Output: u(x, t) solution
    
    Architecture: Fully connected network with residual connections
    """
    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        activation='tanh'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layer: (x, t) -> hidden_dim
        self.input_layer = nn.Linear(2, hidden_dim)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layer: hidden_dim -> u
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights (Xavier for tanh)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better gradient flow with tanh."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, xt):
        """
        Forward pass.
        
        Args:
            xt: (B, N, 2) or (N, 2) - coordinates [x, t]
        
        Returns:
            u: (B, N, 1) or (N, 1) - solution values
        """
        # Normalize input to [-1, 1] for better training
        # Assuming x, t in [0, 1]
        xt_norm = 2 * xt - 1  # [0,1] -> [-1,1]
        
        # Input layer
        h = self.activation(self.input_layer(xt_norm))
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            h_new = self.activation(layer(h))
            h = h + h_new  # Residual connection
        
        # Output layer
        u = self.output_layer(h)
        
        return u


# ============================================================================
# PDE Residual
# ============================================================================

def compute_pde_residual(model, xt, nu):
    """
    Compute PDE residual: u_t - nu * u_xx = 0
    
    Args:
        model: PINN model
        xt: (B, N, 2) - coordinates with requires_grad=True
        nu: diffusivity coefficient
    
    Returns:
        residual: (B, N, 1)
    """
    # Ensure gradients are enabled
    xt.requires_grad_(True)
    
    # Forward pass
    u = model(xt)
    
    # First derivatives
    grads = torch.autograd.grad(
        u, xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]  # (B, N, 2)
    
    u_x = grads[..., 0:1]
    u_t = grads[..., 1:2]
    
    # Second derivative u_xx
    u_xx = torch.autograd.grad(
        u_x, xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0][..., 0:1]
    
    # PDE residual
    residual = u_t - nu * u_xx
    
    return residual


# ============================================================================
# Dataset
# ============================================================================

class SparseHeatDataset(Dataset):
    """
    Dataset for sparse heat equation reconstruction.
    Same setup as PAT for fair comparison.
    """
    def __init__(
        self,
        num_instances=10**6,
        M_sparse=200,
        nu=0.1,
        modes=[1, 2, 3],
        Nc=2000,
        Nbc=200,
        Nic=200,
        device='cpu'
    ):
        super().__init__()
        self.num_instances = num_instances
        self.M = M_sparse
        self.nu = nu
        self.modes = modes
        self.device = torch.device(device)
        
        self.Nc = Nc
        self.Nbc = Nbc
        self.Nic = Nic
    
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
        
        # Collocation points for PDE
        xt_colloc = torch.rand(self.Nc, 2, device=self.device)
        
        # Boundary conditions
        t_bc_a = torch.rand(self.Nbc, 1, device=self.device)
        xt_bc_a = torch.cat([torch.zeros_like(t_bc_a), t_bc_a], dim=-1)
        u_bc_a = exact_solution(
            xt_bc_a[:, 0], xt_bc_a[:, 1], self.nu, mode_n
        ).unsqueeze(-1)
        
        t_bc_b = torch.rand(self.Nbc, 1, device=self.device)
        xt_bc_b = torch.cat([torch.ones_like(t_bc_b), t_bc_b], dim=-1)
        u_bc_b = exact_solution(
            xt_bc_b[:, 0], xt_bc_b[:, 1], self.nu, mode_n
        ).unsqueeze(-1)
        
        # Initial condition
        x_ic = torch.rand(self.Nic, 1, device=self.device)
        xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=-1)
        u_ic = exact_solution(
            xt_ic[:, 0], xt_ic[:, 1], self.nu, mode_n
        ).unsqueeze(-1)
        
        return {
            'xt_sparse': xt_sparse,
            'u_sparse': u_sparse,
            'xt_colloc': xt_colloc,
            'xt_bc_a': xt_bc_a,
            'u_bc_a': u_bc_a,
            'xt_bc_b': xt_bc_b,
            'u_bc_b': u_bc_b,
            'xt_ic': xt_ic,
            'u_ic': u_ic,
            'mode': mode_n
        }


# ============================================================================
# Training
# ============================================================================

def train_step(model, optimizer, batch, nu, device, lambda_pde=1.0):
    """Single training step for PINN."""
    model.train()
    optimizer.zero_grad()
    
    # Get data
    xt_sparse = batch['xt_sparse'].to(device)
    u_sparse = batch['u_sparse'].to(device)
    xt_colloc = batch['xt_colloc'].to(device)
    xt_bc_a = batch['xt_bc_a'].to(device)
    u_bc_a = batch['u_bc_a'].to(device)
    xt_bc_b = batch['xt_bc_b'].to(device)
    u_bc_b = batch['u_bc_b'].to(device)
    xt_ic = batch['xt_ic'].to(device)
    u_ic = batch['u_ic'].to(device)
    
    # Data loss: fit sparse observations
    u_pred_sparse = model(xt_sparse)
    loss_data = F.mse_loss(u_pred_sparse, u_sparse)
    
    # PDE residual loss
    residual = compute_pde_residual(model, xt_colloc, nu)
    loss_pde = (residual ** 2).mean()
    
    # Boundary conditions
    u_pred_bc_a = model(xt_bc_a)
    u_pred_bc_b = model(xt_bc_b)
    loss_bc = F.mse_loss(u_pred_bc_a, u_bc_a) + F.mse_loss(u_pred_bc_b, u_bc_b)
    
    # Initial condition
    u_pred_ic = model(xt_ic)
    loss_ic = F.mse_loss(u_pred_ic, u_ic)
    
    # Total loss (weighted)
    loss = loss_data + lambda_pde * (loss_pde + loss_bc + loss_ic)
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'data': loss_data.item(),
        'pde': loss_pde.item(),
        'bc': loss_bc.item(),
        'ic': loss_ic.item()
    }


def evaluate_model(model, nu, mode_n, device, nx=256, nt=128):
    """Evaluate model on fine grid."""
    model.eval()
    
    with torch.no_grad():
        x = torch.linspace(0, 1, nx, device=device)
        t = torch.linspace(0, 1, nt, device=device)
        X, T = torch.meshgrid(x, t, indexing='ij')
        
        xt_eval = torch.stack([X, T], dim=-1).reshape(-1, 2)
        
        # Predict in batches to avoid memory issues
        batch_size = 8192
        u_pred_list = []
        for i in range(0, len(xt_eval), batch_size):
            xt_batch = xt_eval[i:i+batch_size]
            u_batch = model(xt_batch)
            u_pred_list.append(u_batch)
        
        u_pred = torch.cat(u_pred_list, dim=0).reshape(nx, nt)
        u_true = exact_solution(X, T, nu, mode_n)
        
        # Compute errors
        mse = F.mse_loss(u_pred, u_true).item()
        mae = (u_pred - u_true).abs().mean().item()
        
        # Relative error (masked)
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


def plot_results(history, eval_results, save_path='pinn_results.png'):
    """Plot training curves and solution comparison."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(history['step'], history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Loss components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(history['step'], history['data'], label='Data', linewidth=2)
    ax2.semilogy(history['step'], history['pde'], label='PDE', linewidth=2)
    ax2.semilogy(history['step'], history['bc'], label='BC', linewidth=2)
    ax2.semilogy(history['step'], history['ic'], label='IC', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Evaluation metrics
    ax3 = fig.add_subplot(gs[0, 2])
    if len(history['eval_mse']) > 0:
        ax3.semilogy(history['eval_step'], history['eval_mse'], 'ro-', label='MSE')
        ax3.semilogy(history['eval_step'], history['eval_mae'], 'go-', label='MAE')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Error')
        ax3.set_title('Evaluation Metrics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Ground truth
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.contourf(eval_results['T'], eval_results['X'], 
                      eval_results['ground_truth'], levels=50, cmap='viridis')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Space x')
    ax4.set_title('Ground Truth')
    plt.colorbar(im, ax=ax4)
    
    # Prediction
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.contourf(eval_results['T'], eval_results['X'],
                      eval_results['prediction'], levels=50, cmap='viridis')
    ax5.set_xlabel('Time t')
    ax5.set_ylabel('Space x')
    ax5.set_title('PINN Prediction')
    plt.colorbar(im, ax=ax5)
    
    # Error
    ax6 = fig.add_subplot(gs[1, 2])
    error = (eval_results['prediction'] - eval_results['ground_truth']).abs()
    im = ax6.contourf(eval_results['T'], eval_results['X'], 
                      error, levels=50, cmap='Reds')
    ax6.set_xlabel('Time t')
    ax6.set_ylabel('Space x')
    ax6.set_title(f'Absolute Error (max={error.max():.2e})')
    plt.colorbar(im, ax=ax6)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to {save_path}")


# ============================================================================
# Main Training Function
# ============================================================================

def train_pinn(args):
    """Main training loop for PINN."""
    device = torch.device(args.device)
    print(f"\n{'='*80}")
    print(f"PINN - Sparse Reconstruction Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Sparse points (M): {args.M}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Learning rate: {args.lr}")
    print(f"PDE weight: {args.lambda_pde}")
    print(f"{'='*80}\n")
    
    # Create dataset
    dataset = SparseHeatDataset(
        M_sparse=args.M,
        nu=args.nu,
        modes=args.modes,
        Nc=args.Nc,
        Nbc=args.Nbc,
        Nic=args.Nic,
        device=device
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    model = PINN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.lr * 0.01
    )
    
    # Training history
    history = {
        'step': [], 'loss': [], 'data': [], 'pde': [], 'bc': [], 'ic': [],
        'eval_step': [], 'eval_mse': [], 'eval_mae': [], 'eval_rel': []
    }
    
    # Training loop
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
        
        # Training step
        stats = train_step(model, optimizer, batch, args.nu, device, args.lambda_pde)
        scheduler.step()
        
        # Logging
        if step % args.print_every == 0 or step == 1:
            lr_current = scheduler.get_last_lr()[0]
            print(f"[{step:05d}] loss={stats['loss']:.3e} | "
                  f"data={stats['data']:.3e} pde={stats['pde']:.3e} "
                  f"bc={stats['bc']:.3e} ic={stats['ic']:.3e} | "
                  f"lr={lr_current:.2e}")
            
            history['step'].append(step)
            history['loss'].append(stats['loss'])
            history['data'].append(stats['data'])
            history['pde'].append(stats['pde'])
            history['bc'].append(stats['bc'])
            history['ic'].append(stats['ic'])
        
        # Evaluation
        if step % args.eval_every == 0 or step == args.steps:
            eval_results = evaluate_model(model, args.nu, args.modes[0], device)
            
            print(f"       EVAL → MSE={eval_results['mse']:.3e} "
                  f"MAE={eval_results['mae']:.3e} "
                  f"RelErr={eval_results['rel_error']:.2%}")
            
            history['eval_step'].append(step)
            history['eval_mse'].append(eval_results['mse'])
            history['eval_mae'].append(eval_results['mae'])
            history['eval_rel'].append(eval_results['rel_error'])
            
            if eval_results['mse'] < best_mse:
                best_mse = eval_results['mse']
                print(f"       → New best MSE: {best_mse:.3e}")
                
                # Save best model
                if args.save_path:
                    best_path = args.save_path.replace('.pt', '_best.pt')
                    os.makedirs(os.path.dirname(best_path), exist_ok=True)
                    torch.save({
                        'model': model.state_dict(),
                        'step': step,
                        'mse': best_mse,
                        'args': vars(args)
                    }, best_path)
            
            print("-" * 80)
        
        # Save checkpoint
        if step % args.save_every == 0 and args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step,
                'history': history,
                'args': vars(args)
            }, args.save_path)
    
    # Final evaluation
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    final_eval = evaluate_model(model, args.nu, args.modes[0], device)
    print(f"Final MSE:        {final_eval['mse']:.4e}")
    print(f"Final MAE:        {final_eval['mae']:.4e}")
    print(f"Final Rel Error:  {final_eval['rel_error']:.2%}")
    print(f"Max Error:        {final_eval['max_error']:.4e}")
    print(f"Training time:    {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Best MSE:         {best_mse:.4e}")
    print("="*80)
    
    # Generate plots
    if args.plot:
        plot_results(history, final_eval, 'pinn_results.png')
    
    # Save final model
    if args.save_path:
        final_path = args.save_path.replace('.pt', '_final.pt')
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'step': args.steps,
            'mse': final_eval['mse'],
            'history': history,
            'args': vars(args)
        }, final_path)
        print(f"\nFinal model saved to {final_path}")
    
    return model, history, final_eval


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PINN for sparse heat equation reconstruction")
    
    # Problem setup
    parser.add_argument("--nu", type=float, default=0.1, help="Thermal diffusivity")
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2, 3], help="Fourier modes")
    parser.add_argument("--M", type=int, default=200, help="Number of sparse observations")
    
    # Sampling
    parser.add_argument("--Nc", type=int, default=2000, help="Collocation points")
    parser.add_argument("--Nbc", type=int, default=200, help="Boundary points")
    parser.add_argument("--Nic", type=int, default=200, help="Initial condition points")
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--activation", type=str, default='tanh', 
                       choices=['tanh', 'relu', 'gelu'], help="Activation function")
    
    # Training
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--lambda_pde", type=float, default=1.0, help="PDE loss weight")
    
    # Logging
    parser.add_argument("--print_every", type=int, default=50, help="Print frequency")
    parser.add_argument("--eval_every", type=int, default=200, help="Eval frequency")
    parser.add_argument("--save_every", type=int, default=1000, help="Save frequency")
    parser.add_argument("--save_path", type=str, default="checkpoints/pinn_sparse.pt")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pinn(args)