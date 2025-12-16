#!/usr/bin/env python3
"""
SIREN (Sinusoidal Representation Networks) for Sparse Reconstruction
====================================================================

Implements SIREN with periodic sine activations for 1D heat equation.
Uses special initialization scheme for implicit neural representations.

Reference: Sitzmann et al. "Implicit Neural Representations with Periodic 
           Activation Functions" (NeurIPS 2020)

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
    """Exact solution for 1D heat equation."""
    return torch.exp(-nu * (n * math.pi)**2 * t) * torch.sin(n * math.pi * x)


# ============================================================================
# SIREN Layers
# ============================================================================

class SineLayer(nn.Module):
    """
    Single SIREN layer with sine activation.
    
    Uses special initialization based on:
    - First layer: uniform(-1/n, 1/n)
    - Hidden layers: uniform(-sqrt(6/n)/w0, sqrt(6/n)/w0)
    """
    def __init__(self, in_features, out_features, bias=True, 
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self):
        """Special initialization for SIREN."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform(-1/n, 1/n)
                self.linear.weight.uniform_(-1 / self.in_features, 
                                           1 / self.in_features)
            else:
                # Hidden layers: uniform(-sqrt(6/n)/w0, sqrt(6/n)/w0)
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN network with sine activations throughout.
    
    Architecture: All layers use sine activation with special initialization.
    """
    def __init__(
        self,
        hidden_dim=256,
        num_layers=5,
        omega_0=30.0,
        omega_hidden=30.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # First layer (special omega_0)
        self.first_layer = SineLayer(
            2, hidden_dim, 
            is_first=True, 
            omega_0=omega_0
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            SineLayer(hidden_dim, hidden_dim, 
                     is_first=False, 
                     omega_0=omega_hidden)
            for _ in range(num_layers - 1)
        ])
        
        # Output layer (linear, no sine activation)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize output layer
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / omega_hidden
            self.output_layer.weight.uniform_(-bound, bound)
    
    def forward(self, xt):
        """
        Forward pass.
        
        Args:
            xt: (B, N, 2) or (N, 2) - coordinates [x, t]
        
        Returns:
            u: (B, N, 1) or (N, 1) - solution values
        """
        # Input normalization (optional, SIREN often works without)
        # xt is assumed to be in [0, 1], which is fine for SIREN
        
        # First layer
        h = self.first_layer(xt)
        
        # Hidden layers
        for layer in self.hidden_layers:
            h = layer(h)
        
        # Output layer (no activation)
        u = self.output_layer(h)
        
        return u


# ============================================================================
# PDE Residual
# ============================================================================

def compute_pde_residual(model, xt, nu):
    """
    Compute PDE residual: u_t - nu * u_xx = 0
    """
    xt.requires_grad_(True)
    u = model(xt)
    
    # First derivatives
    grads = torch.autograd.grad(
        u, xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]
    
    u_x = grads[..., 0:1]
    u_t = grads[..., 1:2]
    
    # Second derivative
    u_xx = torch.autograd.grad(
        u_x, xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0][..., 0:1]
    
    residual = u_t - nu * u_xx
    return residual


# ============================================================================
# Dataset (Same as PINN)
# ============================================================================

class SparseHeatDataset(Dataset):
    """Dataset for sparse heat equation reconstruction."""
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
        
        # Collocation points
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
    """Single training step."""
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
    
    # Data loss
    u_pred_sparse = model(xt_sparse)
    loss_data = F.mse_loss(u_pred_sparse, u_sparse)
    
    # PDE residual
    residual = compute_pde_residual(model, xt_colloc, nu)
    loss_pde = (residual ** 2).mean()
    
    # Boundary conditions
    u_pred_bc_a = model(xt_bc_a)
    u_pred_bc_b = model(xt_bc_b)
    loss_bc = F.mse_loss(u_pred_bc_a, u_bc_a) + F.mse_loss(u_pred_bc_b, u_bc_b)
    
    # Initial condition
    u_pred_ic = model(xt_ic)
    loss_ic = F.mse_loss(u_pred_ic, u_ic)
    
    # Total loss
    loss = loss_data + lambda_pde * (loss_pde + loss_bc + loss_ic)
    
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
        
        # Predict in batches
        batch_size = 8192
        u_pred_list = []
        for i in range(0, len(xt_eval), batch_size):
            xt_batch = xt_eval[i:i+batch_size]
            u_batch = model(xt_batch)
            u_pred_list.append(u_batch)
        
        u_pred = torch.cat(u_pred_list, dim=0).reshape(nx, nt)
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


def plot_results(history, eval_results, save_path='siren_results.png'):
    """Plot training and results."""
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
    
    # Evaluation
    ax3 = fig.add_subplot(gs[0, 2])
    if len(history['eval_mse']) > 0:
        ax3.semilogy(history['eval_step'], history['eval_mse'], 'b-o', label='Train MSE', markersize=4)
        ax3.semilogy(history['eval_step'], history['eval_mae'], 'b--s', label='Train MAE', markersize=4)
        if len(history.get('test_mse', [])) > 0:
            ax3.semilogy(history['test_step'], history['test_mse'], 'r-o', label='Test MSE', markersize=4)
            ax3.semilogy(history['test_step'], history['test_mae'], 'r--s', label='Test MAE', markersize=4)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Error')
        ax3.set_title('Train vs Test Error')
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
    ax5.set_title('SIREN Prediction')
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
# Main Training
# ============================================================================

def train_siren(args):
    """Main training loop for SIREN."""
    device = torch.device(args.device)
    print(f"\n{'='*80}")
    print(f"SIREN - Sparse Reconstruction Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Sparse points (M): {args.M}")
    print(f"Training modes: {args.train_modes}")
    print(f"Test modes: {args.test_modes}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Omega_0: {args.omega_0}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*80}\n")
    
    # Dataset
    dataset = SparseHeatDataset(
        M_sparse=args.M,
        nu=args.nu,
        modes=args.train_modes,
        Nc=args.Nc,
        Nbc=args.Nbc,
        Nic=args.Nic,
        device=device
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = SIREN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=args.omega_0,
        omega_hidden=args.omega_hidden
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    # Optimizer (Adam works well with SIREN)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    
    # History
    history = {
        'step': [], 'loss': [], 'data': [], 'pde': [], 'bc': [], 'ic': [],
        'eval_step': [], 'eval_mse': [], 'eval_mae': [], 'eval_rel': [],
        'test_step': [], 'test_mse': [], 'test_mae': [], 'test_rel': []
    }
    
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
        
        stats = train_step(model, optimizer, batch, args.nu, device, args.lambda_pde)
        scheduler.step()
        
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
        
        if step % args.eval_every == 0 or step == args.steps:
            eval_results = evaluate_model(model, args.nu, args.train_modes[0], device)
            
            print(f"       EVAL (seen mode {args.train_modes[0]}) → MSE={eval_results['mse']:.3e} "
                  f"MAE={eval_results['mae']:.3e} "
                  f"RelErr={eval_results['rel_error']:.2%}")
            
            history['eval_step'].append(step)
            history['eval_mse'].append(eval_results['mse'])
            history['eval_mae'].append(eval_results['mae'])
            history['eval_rel'].append(eval_results['rel_error'])
            
            # Test set evaluation
            test_results = evaluate_model(model, args.nu, args.test_modes[0], device)
            
            print(f"       TEST (unseen mode {args.test_modes[0]}) → MSE={test_results['mse']:.3e} "
                  f"MAE={test_results['mae']:.3e} "
                  f"RelErr={test_results['rel_error']:.2%}")
            
            history['test_step'].append(step)
            history['test_mse'].append(test_results['mse'])
            history['test_mae'].append(test_results['mae'])
            history['test_rel'].append(test_results['rel_error'])
            
            if eval_results['mse'] < best_mse:
                best_mse = eval_results['mse']
                print(f"       → New best MSE: {best_mse:.3e}")
                
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
    print("FINAL EVALUATION")
    print("="*80)
    
    print("\n--- Interpolation (Seen Modes) ---")
    for mode in args.train_modes:
        eval_seen = evaluate_model(model, args.nu, mode, device)
        print(f"Mode {mode}: MSE={eval_seen['mse']:.4e}, MAE={eval_seen['mae']:.4e}, "
              f"RelErr={eval_seen['rel_error']:.4e}")
    
    print("\n--- Extrapolation (Unseen Modes) ---")
    for mode in args.test_modes:
        eval_unseen = evaluate_model(model, args.nu, mode, device)
        print(f"Mode {mode}: MSE={eval_unseen['mse']:.4e}, MAE={eval_unseen['mae']:.4e}, "
              f"RelErr={eval_unseen['rel_error']:.4e}")
    
    print(f"\nTraining time:    {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Best MSE:         {best_mse:.4e}")
    print("="*80)
    
    if args.plot:
        final_eval = evaluate_model(model, args.nu, args.train_modes[0], device)
        plot_results(history, final_eval, 'siren_seen_mode.png')
        
        unseen_eval = evaluate_model(model, args.nu, args.test_modes[0], device)
        plot_results(history, unseen_eval, 'siren_unseen_mode.png')
    
    if args.save_path:
        final_path = args.save_path.replace('.pt', '_final.pt')
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        final_eval = evaluate_model(model, args.nu, args.train_modes[0], device)
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
    parser = argparse.ArgumentParser(description="SIREN for sparse heat equation")
    
    # Problem
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--train_modes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--test_modes", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--M", type=int, default=200)
    
    # Sampling
    parser.add_argument("--Nc", type=int, default=2000)
    parser.add_argument("--Nbc", type=int, default=200)
    parser.add_argument("--Nic", type=int, default=200)
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--omega_0", type=float, default=30.0)
    parser.add_argument("--omega_hidden", type=float, default=30.0)
    
    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_pde", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="checkpoints/siren_sparse.pt")
    parser.add_argument("--plot", action="store_true", default=True)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_siren(args)