#!/usr/bin/env python3
"""
Load and evaluate trained SIREN model for Navier-Stokes

# Evaluate SIREN model
python evaluate_siren.py --checkpoint checkpoints/siren_ns.pt

# With custom parameters
python evaluate_siren.py \
    --checkpoint checkpoints/siren_ns.pt \
    --data_path ./cylinder_nektar_wake.mat \
    --t_index 100 \
    --Re 100.0 \
    --hidden_dim 256 \
    --n_layers 4 \
    --omega_0 30.0 \
    --device cuda

"""

import os
import math
import argparse
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# Model Definition (same as training script)
# =============================================================================

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / in_features) / omega_0,
                    math.sqrt(6 / in_features) / omega_0
                )
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=4, 
                 out_features=3, omega_0=30.0):
        super().__init__()
        
        self.net = nn.ModuleList()
        self.net.append(SIRENLayer(in_features, hidden_features, 
                                   omega_0=omega_0, is_first=True))
        
        for _ in range(hidden_layers):
            self.net.append(SIRENLayer(hidden_features, hidden_features, 
                                      omega_0=omega_0, is_first=False))
        
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -math.sqrt(6 / hidden_features) / omega_0,
                math.sqrt(6 / hidden_features) / omega_0
            )
        self.net.append(final_linear)
    
    def forward(self, coords):
        """coords: (B, N, 3) -> (x, y, t)"""
        x = coords
        for i, layer in enumerate(self.net[:-1]):
            x = layer(x)
        x = self.net[-1](x)
        return x


# =============================================================================
# Data Loader
# =============================================================================

class CylinderWakeData:
    def __init__(self, mat_path, seed=0):
        data = scipy.io.loadmat(mat_path)
        self.U_star = data["U_star"]
        self.p_star = data["p_star"]
        self.X_star = data["X_star"]
        self.t_star = data["t"]
        
        self.N = self.X_star.shape[0]
        self.T = self.t_star.shape[0]
        
        # Flatten
        XX = np.tile(self.X_star[:, 0:1], (1, self.T))
        YY = np.tile(self.X_star[:, 1:2], (1, self.T))
        TT = np.tile(self.t_star, (1, self.N)).T
        
        self.x = XX.flatten()[:, None]
        self.y = YY.flatten()[:, None]
        self.t = TT.flatten()[:, None]
        self.u = self.U_star[:, 0, :].flatten()[:, None]
        self.v = self.U_star[:, 1, :].flatten()[:, None]
        self.p = self.p_star.flatten()[:, None]
        
        self.NT = self.x.shape[0]
        self.rng = np.random.RandomState(seed)
    
    def get_snapshot(self, t_idx):
        x = self.X_star[:, 0:1]
        y = self.X_star[:, 1:2]
        t = np.full_like(x, self.t_star[t_idx, 0])
        u = self.U_star[:, 0, t_idx:t_idx+1]
        v = self.U_star[:, 1, t_idx:t_idx+1]
        p = self.p_star[:, t_idx:t_idx+1]
        xyt = np.concatenate([x, y, t], axis=1)
        uvp = np.concatenate([u, v, p], axis=1)
        return xyt, uvp


# =============================================================================
# NS Residuals
# =============================================================================

def ns_residuals(uvp, xyt, nu):
    """Compute Navier-Stokes residuals"""
    u = uvp[..., 0:1]
    v = uvp[..., 1:2]
    p = uvp[..., 2:3]
    
    grads_u = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
    grads_v = torch.autograd.grad(v, xyt, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
    grads_p = torch.autograd.grad(p, xyt, grad_outputs=torch.ones_like(p),
                                  create_graph=True, retain_graph=True)[0]
    
    u_x, u_y, u_t = grads_u[..., 0:1], grads_u[..., 1:2], grads_u[..., 2:3]
    v_x, v_y, v_t = grads_v[..., 0:1], grads_v[..., 1:2], grads_v[..., 2:3]
    p_x, p_y = grads_p[..., 0:1], grads_p[..., 1:2]
    
    u_xx = torch.autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True)[0][..., 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, grad_outputs=torch.ones_like(u_y),
                              create_graph=True, retain_graph=True)[0][..., 1:2]
    v_xx = torch.autograd.grad(v_x, xyt, grad_outputs=torch.ones_like(v_x),
                              create_graph=True, retain_graph=True)[0][..., 0:1]
    v_yy = torch.autograd.grad(v_y, xyt, grad_outputs=torch.ones_like(v_y),
                              create_graph=True, retain_graph=True)[0][..., 1:2]
    
    r_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    r_c = u_x + v_y
    
    return r_u, r_v, r_c


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_psnr(pred, true):
    """
    Compute PSNR between prediction and ground truth
    
    Args:
        pred: predicted values (numpy array)
        true: ground truth values (numpy array)
    
    Returns:
        psnr: PSNR in dB
    """
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()
    
    mse = np.mean((pred - true) ** 2)
    if mse < 1e-12:
        return np.inf
    
    # Use data range (max - min)
    data_range = np.max(true) - np.min(true)
    
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_comprehensive_metrics(model, data, device, t_index, nu):
    """
    Compute comprehensive metrics including PSNR, Rel-L2, MSE, and PDE residuals
    
    Args:
        model: Trained SIREN model
        data: CylinderWakeData instance
        device: torch device
        t_index: time snapshot index
        nu: kinematic viscosity
    
    Returns:
        dict with all metrics
    """
    model.eval()
    
    # Get test snapshot
    xyt_test, uvp_test = data.get_snapshot(t_index)
    xyt_test_torch = torch.tensor(xyt_test, dtype=torch.float32, device=device)
    uvp_test_torch = torch.tensor(uvp_test, dtype=torch.float32, device=device)
    
    # Prediction
    with torch.no_grad():
        uvp_pred_torch = model(xyt_test_torch)
    
    # Convert to numpy
    uvp_pred = uvp_pred_torch.cpu().numpy()
    
    # Extract individual components
    u_true = uvp_test[:, 0].flatten()
    v_true = uvp_test[:, 1].flatten()
    p_true = uvp_test[:, 2].flatten()
    
    u_pred = uvp_pred[:, 0].flatten()
    v_pred = uvp_pred[:, 1].flatten()
    p_pred = uvp_pred[:, 2].flatten()
    
    # MSE
    mse_u = np.mean((u_pred - u_true) ** 2)
    mse_v = np.mean((v_pred - v_true) ** 2)
    mse_p = np.mean((p_pred - p_true) ** 2)
    mse_total = (mse_u + mse_v + mse_p) / 3.0
    
    # Relative L2 Error
    rel_l2_u = np.linalg.norm(u_pred - u_true, 2) / np.linalg.norm(u_true, 2)
    rel_l2_v = np.linalg.norm(v_pred - v_true, 2) / np.linalg.norm(v_true, 2)
    rel_l2_p = np.linalg.norm(p_pred - p_true, 2) / np.linalg.norm(p_true, 2)
    rel_l2_avg = (rel_l2_u + rel_l2_v + rel_l2_p) / 3.0
    
    # PSNR
    psnr_u = compute_psnr(u_pred, u_true)
    psnr_v = compute_psnr(v_pred, v_true)
    psnr_p = compute_psnr(p_pred, p_true)
    psnr_avg = (psnr_u + psnr_v + psnr_p) / 3.0
    
    # PDE Residuals
    with torch.enable_grad():
        xyt_test_grad = xyt_test_torch.clone().requires_grad_(True)
        uvp_pred_grad = model(xyt_test_grad)
        r_u, r_v, r_c = ns_residuals(uvp_pred_grad, xyt_test_grad, nu)
        
        pde_residual = (r_u.square().mean() + r_v.square().mean() + r_c.square().mean()).item()
    
    return {
        'mse_u': mse_u,
        'mse_v': mse_v,
        'mse_p': mse_p,
        'mse_total': mse_total,
        'rel_l2_u': rel_l2_u,
        'rel_l2_v': rel_l2_v,
        'rel_l2_p': rel_l2_p,
        'rel_l2_avg': rel_l2_avg,
        'psnr_u': psnr_u,
        'psnr_v': psnr_v,
        'psnr_p': psnr_p,
        'psnr_avg': psnr_avg,
        'pde_residual': pde_residual,
        'u_pred': u_pred,
        'v_pred': v_pred,
        'p_pred': p_pred,
        'u_true': u_true,
        'v_true': v_true,
        'p_true': p_true,
        'x': xyt_test[:, 0],
        'y': xyt_test[:, 1],
        't': data.t_star[t_index, 0],
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_siren_model(checkpoint_path, device, hidden_dim=256, n_layers=4, omega_0=30.0):
    """
    Load trained SIREN model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        hidden_dim: Hidden layer dimension (default: 256)
        n_layers: Number of hidden layers (default: 4)
        omega_0: SIREN frequency (default: 30.0)
    
    Returns:
        model: Loaded SIREN model
        checkpoint: Checkpoint dictionary
    """
    print(f"\nLoading SIREN model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = SIREN(
        in_features=3,
        hidden_features=hidden_dim,
        hidden_layers=n_layers,
        out_features=3,
        omega_0=omega_0
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"  Training step: {checkpoint.get('step', 'N/A')}")
    print(f"  Test MSE: {checkpoint.get('test_mse', 'N/A'):.6e}")
    
    return model, checkpoint


# =============================================================================
# Visualization
# =============================================================================

def plot_evaluation_results(metrics, save_dir='./siren_evaluation'):
    """Create comprehensive visualization of evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    x = metrics['x']
    y = metrics['y']
    
    # Detect grid structure
    xu = np.unique(np.round(x, 10))
    yu = np.unique(np.round(y, 10))
    nx, ny = len(xu), len(yu)
    
    is_grid = (len(x) == nx * ny)
    
    if is_grid:
        x_to_i = {val: i for i, val in enumerate(xu)}
        y_to_j = {val: j for j, val in enumerate(yu)}
        
        def to_grid(values):
            grid = np.full((ny, nx), np.nan)
            for xi, yi, val in zip(np.round(x, 10), np.round(y, 10), values):
                grid[y_to_j[yi], x_to_i[xi]] = val
            return grid
        
        extent = [xu.min(), xu.max(), yu.min(), yu.max()]
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    variables = ['u', 'v', 'p']
    var_names = ['u-velocity', 'v-velocity', 'pressure']
    
    for i, (var, var_name) in enumerate(zip(variables, var_names)):
        true_val = metrics[f'{var}_true']
        pred_val = metrics[f'{var}_pred']
        err_val = np.abs(pred_val - true_val)
        rel_err_val = err_val / (np.abs(true_val) + 1e-10)
        
        # True field
        ax = fig.add_subplot(gs[i, 0])
        if is_grid:
            grid = to_grid(true_val)
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent,
                          cmap='RdBu_r', interpolation='bilinear')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            im = ax.scatter(x, y, c=true_val, s=2, cmap='RdBu_r')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        ax.set_title(f'{var_name} (Ground Truth)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Predicted field
        ax = fig.add_subplot(gs[i, 1])
        if is_grid:
            grid = to_grid(pred_val)
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent,
                          cmap='RdBu_r', interpolation='bilinear')
            ax.set_xlabel('x')
        else:
            im = ax.scatter(x, y, c=pred_val, s=2, cmap='RdBu_r')
            ax.set_xlabel('x')
            ax.set_aspect('equal')
        ax.set_title(f'{var_name} (Prediction)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Absolute error
        ax = fig.add_subplot(gs[i, 2])
        if is_grid:
            grid = to_grid(err_val)
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent,
                          cmap='hot', interpolation='bilinear')
            ax.set_xlabel('x')
        else:
            im = ax.scatter(x, y, c=err_val, s=2, cmap='hot')
            ax.set_xlabel('x')
            ax.set_aspect('equal')
        ax.set_title(f'{var_name} (Abs Error)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Relative error
        ax = fig.add_subplot(gs[i, 3])
        if is_grid:
            grid = to_grid(rel_err_val)
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent,
                          cmap='hot', interpolation='bilinear', vmax=0.5)
            ax.set_xlabel('x')
        else:
            im = ax.scatter(x, y, c=np.clip(rel_err_val, 0, 0.5), s=2, cmap='hot')
            ax.set_xlabel('x')
            ax.set_aspect('equal')
        ax.set_title(f'{var_name} (Rel Error)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add metrics text
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    
    metrics_text = f"""
SIREN EVALUATION METRICS (t={metrics['t']:.3f})
{'='*80}

Mean Squared Error (MSE):
  u-velocity: {metrics['mse_u']:.6e}
  v-velocity: {metrics['mse_v']:.6e}
  pressure:   {metrics['mse_p']:.6e}
  Average:    {metrics['mse_total']:.6e}

Relative L2 Error:
  u-velocity: {metrics['rel_l2_u']:.6f}
  v-velocity: {metrics['rel_l2_v']:.6f}
  pressure:   {metrics['rel_l2_p']:.6f}
  Average:    {metrics['rel_l2_avg']:.6f}

Peak Signal-to-Noise Ratio (PSNR):
  u-velocity: {metrics['psnr_u']:.2f} dB
  v-velocity: {metrics['psnr_v']:.2f} dB
  pressure:   {metrics['psnr_p']:.2f} dB
  Average:    {metrics['psnr_avg']:.2f} dB

PDE Residual: {metrics['pde_residual']:.6e}
    """
    
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('SIREN Model Evaluation Results', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved evaluation plot to: {save_path}")


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_siren(checkpoint_path, data_path, t_index=100, Re=100.0, 
                   hidden_dim=256, n_layers=4, omega_0=30.0, device='cuda'):
    """
    Complete evaluation pipeline for SIREN model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_siren_model(checkpoint_path, device, 
                                        hidden_dim, n_layers, omega_0)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    data = CylinderWakeData(data_path, seed=0)
    print(f"  Spatial points: {data.N}")
    print(f"  Time steps: {data.T}")
    
    # Calculate nu
    nu = 1.0 / Re
    print(f"\nReynolds number: {Re} -> nu={nu:.6f}")
    
    # Compute metrics
    print(f"\nComputing comprehensive metrics at t_index={t_index}...")
    metrics = compute_comprehensive_metrics(model, data, device, t_index, nu)
    
    # Print results
    print(f"\n{'='*80}")
    print("SIREN EVALUATION METRICS")
    print(f"{'='*80}")
    print(f"Time: t={metrics['t']:.3f}")
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
    
    print(f"\nPDE Residual: {metrics['pde_residual']:.6e}")
    print(f"{'='*80}\n")
    
    # Create visualizations
    save_dir = os.path.join(os.path.dirname(checkpoint_path), 'siren_evaluation')
    plot_evaluation_results(metrics, save_dir)
    
    # Save metrics to checkpoint
    save_metrics = input("Save metrics to checkpoint? (y/n): ")
    if save_metrics.lower() == 'y':
        checkpoint['evaluation_metrics'] = metrics
        torch.save(checkpoint, checkpoint_path)
        print(f"Metrics saved to: {checkpoint_path}")
    
    return model, metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SIREN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat',
                       help='Path to data file')
    parser.add_argument('--t_index', type=int, default=100,
                       help='Time index for evaluation')
    parser.add_argument('--Re', type=float, default=100.0,
                       help='Reynolds number')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of hidden layers')
    parser.add_argument('--omega_0', type=float, default=30.0,
                       help='SIREN frequency parameter')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluate_siren(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        t_index=args.t_index,
        Re=args.Re,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        omega_0=args.omega_0,
        device=args.device
    )


if __name__ == '__main__':
    main()