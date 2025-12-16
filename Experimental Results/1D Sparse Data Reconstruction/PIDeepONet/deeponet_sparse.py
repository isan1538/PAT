#!/usr/bin/env python3
"""
Physics-Informed DeepONet for Sparse Reconstruction
===================================================

Implements Physics-Informed Deep Operator Network for 1D heat equation.
Learns the operator from sparse observations to full solution.

Reference: Wang et al. "Learning the solution operator of parametric partial 
           differential equations with physics-informed DeepONets" (2021)

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
# DeepONet Architecture
# ============================================================================

class BranchNet(nn.Module):
    """
    Branch network: encodes the input function (sparse observations).
    
    Input: Sensor readings at fixed locations
    Output: Encoding vector
    """
    def __init__(self, sensor_dim, hidden_dim=128, num_layers=4, output_dim=128):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(sensor_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, u_sensors):
        """
        Args:
            u_sensors: (batch, sensor_dim) - values at sensor locations
        Returns:
            branch_output: (batch, output_dim)
        """
        return self.network(u_sensors)


class TrunkNet(nn.Module):
    """
    Trunk network: encodes the query locations.
    
    Input: Query coordinates (x, t)
    Output: Basis functions
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4, output_dim=128):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, xt):
        """
        Args:
            xt: (batch, n_points, 2) - query coordinates
        Returns:
            trunk_output: (batch, n_points, output_dim)
        """
        return self.network(xt)


class DeepONet(nn.Module):
    """
    Deep Operator Network.
    
    Combines branch and trunk networks via dot product:
    u(x,t) = b₀ + Σᵢ branch_i * trunk_i(x,t)
    """
    def __init__(
        self,
        sensor_dim,
        branch_hidden=128,
        trunk_hidden=128,
        num_layers=4,
        latent_dim=128
    ):
        super().__init__()
        
        self.branch = BranchNet(
            sensor_dim=sensor_dim,
            hidden_dim=branch_hidden,
            num_layers=num_layers,
            output_dim=latent_dim
        )
        
        self.trunk = TrunkNet(
            input_dim=2,  # (x, t)
            hidden_dim=trunk_hidden,
            num_layers=num_layers,
            output_dim=latent_dim
        )
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, u_sensors, xt_query):
        """
        Args:
            u_sensors: (batch, sensor_dim) - sensor readings
            xt_query: (batch, n_query, 2) - query locations
        
        Returns:
            u_pred: (batch, n_query, 1) - predicted values
        """
        # Branch network
        branch_out = self.branch(u_sensors)  # (batch, latent_dim)
        
        # Trunk network
        trunk_out = self.trunk(xt_query)  # (batch, n_query, latent_dim)
        
        # Dot product + bias
        # branch_out: (batch, latent_dim) -> (batch, 1, latent_dim)
        # trunk_out: (batch, n_query, latent_dim)
        u_pred = torch.einsum('bi,bni->bn', branch_out, trunk_out)  # (batch, n_query)
        u_pred = u_pred.unsqueeze(-1) + self.bias  # (batch, n_query, 1)
        
        return u_pred


# ============================================================================
# Physics-Informed DeepONet
# ============================================================================

def compute_pde_residual(model, u_sensors, xt, nu):
    """
    Compute PDE residual for physics-informed training.
    
    Args:
        model: DeepONet model
        u_sensors: (batch, sensor_dim) - sensor readings
        xt: (batch, n_points, 2) - collocation points (requires_grad=True)
        nu: diffusivity
    
    Returns:
        residual: (batch, n_points, 1)
    """
    xt.requires_grad_(True)
    u = model(u_sensors, xt)
    
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
# Dataset with Fixed Sensors
# ============================================================================

class SparseHeatDatasetDeepONet(Dataset):
    """
    Dataset for DeepONet with fixed sensor locations.
    
    DeepONet assumes sensors at fixed locations across all instances.
    We sample the function values at these sensors.
    """
    def __init__(
        self,
        num_instances=10**6,
        M_sensors=200,  # Number of fixed sensor locations
        nu=0.1,
        modes=[1, 2, 3],
        Nc=2000,  # Collocation points for PDE
        Nbc=200,  # Boundary points
        Nic=200,  # Initial condition points
        device='cpu'
    ):
        super().__init__()
        self.num_instances = num_instances
        self.M = M_sensors
        self.nu = nu
        self.modes = modes
        self.device = torch.device(device)
        
        # Fixed sensor locations (same for all instances)
        self.sensor_locations = torch.rand(M_sensors, 2, device=device)
        
        self.Nc = Nc
        self.Nbc = Nbc
        self.Nic = Nic
    
    def __len__(self):
        return self.num_instances
    
    def __getitem__(self, idx):
        mode_n = self.modes[idx % len(self.modes)]
        device = self.device
        
        # Sensor readings at fixed locations
        u_sensors = exact_solution(
            self.sensor_locations[:, 0],
            self.sensor_locations[:, 1],
            self.nu, mode_n
        )  # (M,)
        
        # Random query points for training (data loss)
        n_query = 500
        xt_query = torch.rand(n_query, 2, device=device)
        u_query = exact_solution(
            xt_query[:, 0], xt_query[:, 1],
            self.nu, mode_n
        ).unsqueeze(-1)
        
        # Collocation points for PDE
        xt_colloc = torch.rand(self.Nc, 2, device=device)
        
        # Boundary conditions
        t_bc_a = torch.rand(self.Nbc, 1, device=device)
        xt_bc_a = torch.cat([torch.zeros_like(t_bc_a), t_bc_a], dim=-1)
        u_bc_a = exact_solution(
            xt_bc_a[:, 0], xt_bc_a[:, 1], self.nu, mode_n
        ).unsqueeze(-1)
        
        t_bc_b = torch.rand(self.Nbc, 1, device=device)
        xt_bc_b = torch.cat([torch.ones_like(t_bc_b), t_bc_b], dim=-1)
        u_bc_b = exact_solution(
            xt_bc_b[:, 0], xt_bc_b[:, 1], self.nu, mode_n
        ).unsqueeze(-1)
        
        # Initial condition
        x_ic = torch.rand(self.Nic, 1, device=device)
        xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=-1)
        u_ic = exact_solution(
            xt_ic[:, 0], xt_ic[:, 1], self.nu, mode_n
        ).unsqueeze(-1)
        
        return {
            'u_sensors': u_sensors,
            'xt_query': xt_query,
            'u_query': u_query,
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
    """Single training step for PI-DeepONet."""
    model.train()
    optimizer.zero_grad()
    
    # Get data
    u_sensors = batch['u_sensors'].to(device)
    xt_query = batch['xt_query'].to(device)
    u_query = batch['u_query'].to(device)
    xt_colloc = batch['xt_colloc'].to(device)
    xt_bc_a = batch['xt_bc_a'].to(device)
    u_bc_a = batch['u_bc_a'].to(device)
    xt_bc_b = batch['xt_bc_b'].to(device)
    u_bc_b = batch['u_bc_b'].to(device)
    xt_ic = batch['xt_ic'].to(device)
    u_ic = batch['u_ic'].to(device)
    
    # Data loss: fit query points
    u_pred_query = model(u_sensors, xt_query)
    loss_data = F.mse_loss(u_pred_query, u_query)
    
    # PDE residual loss
    residual = compute_pde_residual(model, u_sensors, xt_colloc, nu)
    loss_pde = (residual ** 2).mean()
    
    # Boundary conditions
    u_pred_bc_a = model(u_sensors, xt_bc_a)
    u_pred_bc_b = model(u_sensors, xt_bc_b)
    loss_bc = F.mse_loss(u_pred_bc_a, u_bc_a) + F.mse_loss(u_pred_bc_b, u_bc_b)
    
    # Initial condition
    u_pred_ic = model(u_sensors, xt_ic)
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


def evaluate_model(model, dataset, nu, mode_n, device, nx=256, nt=128):
    """Evaluate model on fine grid."""
    model.eval()
    
    with torch.no_grad():
        # Get sensor readings for this mode
        u_sensors = exact_solution(
            dataset.sensor_locations[:, 0],
            dataset.sensor_locations[:, 1],
            nu, mode_n
        ).unsqueeze(0)  # (1, M)
        
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
            u_batch = model(u_sensors, xt_batch)
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


def plot_results(history, eval_results, save_path='deeponet_results.png'):
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
    ax5.set_title('PI-DeepONet Prediction')
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

def train_deeponet(args):
    """Main training loop for PI-DeepONet."""
    device = torch.device(args.device)
    print(f"\n{'='*80}")
    print(f"PI-DeepONet - Sparse Reconstruction Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Sensors (M): {args.M}")
    print(f"Branch hidden: {args.branch_hidden}")
    print(f"Trunk hidden: {args.trunk_hidden}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"{'='*80}\n")
    
    # Dataset with fixed sensors
    dataset = SparseHeatDatasetDeepONet(
        M_sensors=args.M,
        nu=args.nu,
        modes=args.modes,
        Nc=args.Nc,
        Nbc=args.Nbc,
        Nic=args.Nic,
        device=device
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = DeepONet(
        sensor_dim=args.M,
        branch_hidden=args.branch_hidden,
        trunk_hidden=args.trunk_hidden,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    
    # History
    history = {
        'step': [], 'loss': [], 'data': [], 'pde': [], 'bc': [], 'ic': [],
        'eval_step': [], 'eval_mse': [], 'eval_mae': [], 'eval_rel': []
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
            eval_results = evaluate_model(model, dataset, args.nu, args.modes[0], device)
            
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
                
                if args.save_path:
                    best_path = args.save_path.replace('.pt', '_best.pt')
                    os.makedirs(os.path.dirname(best_path), exist_ok=True)
                    torch.save({
                        'model': model.state_dict(),
                        'sensor_locations': dataset.sensor_locations,
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
                'sensor_locations': dataset.sensor_locations,
                'step': step,
                'history': history,
                'args': vars(args)
            }, args.save_path)
    
    # Final evaluation
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    final_eval = evaluate_model(model, dataset, args.nu, args.modes[0], device)
    print(f"Final MSE:        {final_eval['mse']:.4e}")
    print(f"Final MAE:        {final_eval['mae']:.4e}")
    print(f"Final Rel Error:  {final_eval['rel_error']:.2%}")
    print(f"Max Error:        {final_eval['max_error']:.4e}")
    print(f"Training time:    {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Best MSE:         {best_mse:.4e}")
    print("="*80)
    
    if args.plot:
        plot_results(history, final_eval, 'deeponet_results.png')
    
    if args.save_path:
        final_path = args.save_path.replace('.pt', '_final.pt')
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'sensor_locations': dataset.sensor_locations,
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
    parser = argparse.ArgumentParser(description="PI-DeepONet for sparse heat equation")
    
    # Problem
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--M", type=int, default=200, help="Number of sensors")
    
    # Sampling
    parser.add_argument("--Nc", type=int, default=2000)
    parser.add_argument("--Nbc", type=int, default=200)
    parser.add_argument("--Nic", type=int, default=200)
    
    # Model
    parser.add_argument("--branch_hidden", type=int, default=128)
    parser.add_argument("--trunk_hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=128)
    
    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_pde", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="checkpoints/deeponet_sparse.pt")
    parser.add_argument("--plot", action="store_true", default=True)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_deeponet(args)
