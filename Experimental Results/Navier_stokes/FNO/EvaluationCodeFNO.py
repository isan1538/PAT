#!/usr/bin/env python3
"""
Load and evaluate trained FNO model for Navier-Stokes


# Evaluate FNO model
python evaluate_fno.py --checkpoint checkpoints/fno_ns.pt

# With custom parameters
python evaluate_fno.py \
    --checkpoint checkpoints/fno_ns.pt \
    --data_path ./cylinder_nektar_wake.mat \
    --t_index 100 \
    --modes 12 \
    --width 32 \
    --n_layers 4 \
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

class SpectralConv2d(nn.Module):
    """2D Fourier layer"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Fourier transform
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], 
                           torch.view_as_complex(self.weights1))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],
                           torch.view_as_complex(self.weights2))
        
        # Inverse Fourier transform
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, n_layers=4, in_channels=3, out_channels=3):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        self.fc0 = nn.Linear(in_channels + 2, self.width)  # +2 for (x,y) encoding
        
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.conv_layers.append(SpectralConv2d(self.width, self.width, 
                                                   self.modes1, self.modes2))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, grid):
        """
        x: (B, H, W, C) input features
        grid: (B, H, W, 2) mesh coordinates
        """
        # Combine input with grid
        x = torch.cat([x, grid], dim=-1)  # (B, H, W, C+2)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (B, width, H, W)
        
        # FNO layers
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
        
        # Output
        x = x.permute(0, 2, 3, 1)  # (B, H, W, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Data Loader
# =============================================================================

class CylinderWakeData:
    def __init__(self, mat_path, nx=100, ny=50):
        data = scipy.io.loadmat(mat_path)
        self.U_star = data["U_star"]  # (N,2,T)
        self.p_star = data["p_star"]  # (N,T)
        self.X_star = data["X_star"]  # (N,2)
        self.t_star = data["t"]       # (T,1)
        
        self.N = self.X_star.shape[0]
        self.T = self.t_star.shape[0]
        self.nx = nx
        self.ny = ny
        
        # Reshape to grid (assuming structured grid)
        self.x = self.X_star[:, 0].reshape(ny, nx)
        self.y = self.X_star[:, 1].reshape(ny, nx)
        
    def get_snapshot(self, t_idx):
        """Get u,v,p at time index t_idx as grid"""
        u = self.U_star[:, 0, t_idx].reshape(self.ny, self.nx)
        v = self.U_star[:, 1, t_idx].reshape(self.ny, self.nx)
        p = self.p_star[:, t_idx].reshape(self.ny, self.nx)
        return u, v, p


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


def compute_comprehensive_metrics(model, data, device, t_index, grid):
    """
    Compute comprehensive metrics including PSNR, Rel-L2, MSE
    
    Args:
        model: Trained FNO model
        data: CylinderWakeData instance
        device: torch device
        t_index: time snapshot index
        grid: (1, H, W, 2) grid coordinates tensor
    
    Returns:
        dict with all metrics
    """
    model.eval()
    
    # Get test snapshot
    u_test, v_test, p_test = data.get_snapshot(t_index)
    uvp_test = np.stack([u_test, v_test, p_test], axis=-1)  # (H, W, 3)
    uvp_test_torch = torch.tensor(uvp_test, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        uvp_pred_torch = model(uvp_test_torch, grid)
    
    # Convert to numpy
    uvp_pred = uvp_pred_torch.squeeze(0).cpu().numpy()  # (H, W, 3)
    
    # Extract individual components
    u_true = uvp_test[:, :, 0].flatten()
    v_true = uvp_test[:, :, 1].flatten()
    p_true = uvp_test[:, :, 2].flatten()
    
    u_pred = uvp_pred[:, :, 0].flatten()
    v_pred = uvp_pred[:, :, 1].flatten()
    p_pred = uvp_pred[:, :, 2].flatten()
    
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
        'u_pred': u_pred.reshape(data.ny, data.nx),
        'v_pred': v_pred.reshape(data.ny, data.nx),
        'p_pred': p_pred.reshape(data.ny, data.nx),
        'u_true': u_test,
        'v_true': v_test,
        'p_true': p_test,
        'x': data.x,
        'y': data.y,
        't': data.t_star[t_index, 0],
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_fno_model(checkpoint_path, device, modes=12, width=32, n_layers=4):
    """
    Load trained FNO model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        modes: Number of Fourier modes (default: 12)
        width: Channel width (default: 32)
        n_layers: Number of layers (default: 4)
    
    Returns:
        model: Loaded FNO model
        checkpoint: Checkpoint dictionary
    """
    print(f"\nLoading FNO model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        n_layers=n_layers,
        in_channels=3,
        out_channels=3
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Test MSE: {checkpoint.get('test_mse', 'N/A'):.6e}")
    
    return model, checkpoint


# =============================================================================
# Visualization
# =============================================================================

def plot_evaluation_results(metrics, save_dir='./fno_evaluation'):
    """Create comprehensive visualization of evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    x = metrics['x']
    y = metrics['y']
    extent = [x.min(), x.max(), y.min(), y.max()]
    
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
        im = ax.imshow(true_val, origin='lower', aspect='auto', extent=extent,
                      cmap='RdBu_r', interpolation='bilinear')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{var_name} (Ground Truth)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Predicted field
        ax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(pred_val, origin='lower', aspect='auto', extent=extent,
                      cmap='RdBu_r', interpolation='bilinear')
        ax.set_xlabel('x')
        ax.set_title(f'{var_name} (Prediction)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Absolute error
        ax = fig.add_subplot(gs[i, 2])
        im = ax.imshow(err_val, origin='lower', aspect='auto', extent=extent,
                      cmap='hot', interpolation='bilinear')
        ax.set_xlabel('x')
        ax.set_title(f'{var_name} (Abs Error)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Relative error
        ax = fig.add_subplot(gs[i, 3])
        im = ax.imshow(rel_err_val, origin='lower', aspect='auto', extent=extent,
                      cmap='hot', interpolation='bilinear', vmax=0.5)
        ax.set_xlabel('x')
        ax.set_title(f'{var_name} (Rel Error)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add metrics text
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    
    metrics_text = f"""
FNO EVALUATION METRICS (t={metrics['t']:.3f})
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
    """
    
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('FNO Model Evaluation Results', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved evaluation plot to: {save_path}")


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_fno(checkpoint_path, data_path, t_index=100, 
                 modes=12, width=32, n_layers=4, device='cuda'):
    """
    Complete evaluation pipeline for FNO model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_fno_model(checkpoint_path, device, modes, width, n_layers)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    data = CylinderWakeData(data_path)
    print(f"  Grid size: {data.ny} x {data.nx}")
    print(f"  Spatial points: {data.N}")
    print(f"  Time steps: {data.T}")
    
    # Create grid
    grid = torch.tensor(np.stack([data.x, data.y], axis=-1), 
                       dtype=torch.float32, device=device).unsqueeze(0)
    
    # Compute metrics
    print(f"\nComputing comprehensive metrics at t_index={t_index}...")
    metrics = compute_comprehensive_metrics(model, data, device, t_index, grid)
    
    # Print results
    print(f"\n{'='*80}")
    print("FNO EVALUATION METRICS")
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
    print(f"{'='*80}\n")
    
    # Create visualizations
    save_dir = os.path.join(os.path.dirname(checkpoint_path), 'fno_evaluation')
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
    parser = argparse.ArgumentParser(description='Evaluate trained FNO model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat',
                       help='Path to data file')
    parser.add_argument('--t_index', type=int, default=100,
                       help='Time index for evaluation')
    parser.add_argument('--modes', type=int, default=12,
                       help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=32,
                       help='Channel width')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of FNO layers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluate_fno(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        t_index=args.t_index,
        modes=args.modes,
        width=args.width,
        n_layers=args.n_layers,
        device=args.device
    )


if __name__ == '__main__':
    main()