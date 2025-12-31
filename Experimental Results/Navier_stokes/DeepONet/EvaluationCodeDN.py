#!/usr/bin/env python3
"""
Load and evaluate trained DeepONet model for Navier-Stokes



# Evaluate DeepONet model
python evaluate_deeponet.py --checkpoint checkpoints/deeponet_ns.pt

# With custom parameters
python evaluate_deeponet.py \
    --checkpoint checkpoints/deeponet_ns.pt \
    --data_path ./cylinder_nektar_wake.mat \
    --t_index 100 \
    --n_sensors 100 \
    --hidden_dim 128 \
    --depth 4 \
    --p 100 \
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

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim=3, hidden_dim=128, 
                 depth=4, p=100, out_dim=3):
        """
        branch_input_dim: number of sensors * channels
        trunk_input_dim: coordinate dimension (x, y, t)
        p: dimension of the inner product space
        out_dim: number of output fields (u, v, p)
        """
        super().__init__()
        self.p = p
        self.out_dim = out_dim
        
        # Branch network (processes sensor data)
        branch_layers = []
        branch_layers.append(nn.Linear(branch_input_dim, hidden_dim))
        branch_layers.append(nn.Tanh())
        for _ in range(depth - 1):
            branch_layers.append(nn.Linear(hidden_dim, hidden_dim))
            branch_layers.append(nn.Tanh())
        branch_layers.append(nn.Linear(hidden_dim, p * out_dim))
        self.branch = nn.Sequential(*branch_layers)
        
        # Trunk network (processes coordinates)
        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_input_dim, hidden_dim))
        trunk_layers.append(nn.Tanh())
        for _ in range(depth - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(hidden_dim, p * out_dim))
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, u_sensors, coords):
        """
        u_sensors: (B, n_sensors * channels) - values at sensor locations
        coords: (B, N, 3) - query coordinates (x, y, t)
        Returns: (B, N, out_dim) - predicted fields
        """
        B, N, _ = coords.shape
        
        # Branch network
        branch_out = self.branch(u_sensors)  # (B, p * out_dim)
        branch_out = branch_out.view(B, self.out_dim, self.p)  # (B, out_dim, p)
        
        # Trunk network
        coords_flat = coords.reshape(B * N, -1)
        trunk_out = self.trunk(coords_flat)  # (B*N, p * out_dim)
        trunk_out = trunk_out.view(B, N, self.out_dim, self.p)  # (B, N, out_dim, p)
        
        # Inner product
        out = torch.einsum('bop,bnop->bno', branch_out, trunk_out)
        out = out + self.bias.view(1, 1, -1)
        
        return out


# =============================================================================
# Data Loader
# =============================================================================

class CylinderWakeData:
    def __init__(self, mat_path, n_sensors=100, seed=0):
        data = scipy.io.loadmat(mat_path)
        self.U_star = data["U_star"]
        self.p_star = data["p_star"]
        self.X_star = data["X_star"]
        self.t_star = data["t"]
        
        self.N = self.X_star.shape[0]
        self.T = self.t_star.shape[0]
        self.n_sensors = n_sensors
        
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
        
        # Fixed sensor locations (randomly selected spatial points)
        self.sensor_idx = self.rng.choice(self.N, n_sensors, replace=False)
    
    def get_snapshot(self, t_idx):
        """Get full snapshot for evaluation"""
        # Sensor values
        u_sensor = self.U_star[self.sensor_idx, 0, t_idx]
        v_sensor = self.U_star[self.sensor_idx, 1, t_idx]
        p_sensor = self.p_star[self.sensor_idx, t_idx]
        sensors = np.concatenate([u_sensor, v_sensor, p_sensor])
        
        # All spatial points at this time
        x = self.X_star[:, 0:1]
        y = self.X_star[:, 1:2]
        t = np.full_like(x, self.t_star[t_idx, 0])
        coords = np.concatenate([x, y, t], axis=1)
        
        u = self.U_star[:, 0, t_idx:t_idx+1]
        v = self.U_star[:, 1, t_idx:t_idx+1]
        p = self.p_star[:, t_idx:t_idx+1]
        vals = np.concatenate([u, v, p], axis=1)
        
        return sensors, coords, vals


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


def compute_comprehensive_metrics(model, data, device, t_index):
    """
    Compute comprehensive metrics including PSNR, Rel-L2, MSE
    
    Args:
        model: Trained DeepONet model
        data: CylinderWakeData instance
        device: torch device
        t_index: time snapshot index
    
    Returns:
        dict with all metrics
    """
    model.eval()
    
    # Get test snapshot
    test_sensors, test_coords, test_vals = data.get_snapshot(t_index)
    test_sensors_torch = torch.tensor(test_sensors, dtype=torch.float32, 
                                     device=device).unsqueeze(0)
    test_coords_torch = torch.tensor(test_coords, dtype=torch.float32, 
                                    device=device).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        uvp_pred_torch = model(test_sensors_torch, test_coords_torch).squeeze(0)
    
    # Convert to numpy
    uvp_pred = uvp_pred_torch.cpu().numpy()
    uvp_true = test_vals
    
    # Extract individual components
    u_true = uvp_true[:, 0].flatten()
    v_true = uvp_true[:, 1].flatten()
    p_true = uvp_true[:, 2].flatten()
    
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
        'u_pred': u_pred,
        'v_pred': v_pred,
        'p_pred': p_pred,
        'u_true': u_true,
        'v_true': v_true,
        'p_true': p_true,
        'x': test_coords[:, 0],
        'y': test_coords[:, 1],
        't': data.t_star[t_index, 0],
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_deeponet_model(checkpoint_path, device, n_sensors=100, 
                        hidden_dim=128, depth=4, p=100):
    """
    Load trained DeepONet model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        n_sensors: Number of sensor points (default: 100)
        hidden_dim: Hidden layer dimension (default: 128)
        depth: Network depth (default: 4)
        p: Inner product dimension (default: 100)
    
    Returns:
        model: Loaded DeepONet model
        checkpoint: Checkpoint dictionary
    """
    print(f"\nLoading DeepONet model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    branch_input_dim = n_sensors * 3  # 3 channels: u, v, p
    model = DeepONet(
        branch_input_dim=branch_input_dim,
        trunk_input_dim=3,
        hidden_dim=hidden_dim,
        depth=depth,
        p=p,
        out_dim=3
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

def plot_evaluation_results(metrics, save_dir='./deeponet_evaluation'):
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
DeepONet EVALUATION METRICS (t={metrics['t']:.3f})
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
    
    fig.suptitle('DeepONet Model Evaluation Results', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved evaluation plot to: {save_path}")


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_deeponet(checkpoint_path, data_path, t_index=100, n_sensors=100,
                      hidden_dim=128, depth=4, p=100, device='cuda'):
    """
    Complete evaluation pipeline for DeepONet model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_deeponet_model(checkpoint_path, device, 
                                           n_sensors, hidden_dim, depth, p)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    data = CylinderWakeData(data_path, n_sensors=n_sensors, seed=0)
    print(f"  Spatial points: {data.N}")
    print(f"  Time steps: {data.T}")
    print(f"  Sensors: {n_sensors}")
    
    # Compute metrics
    print(f"\nComputing comprehensive metrics at t_index={t_index}...")
    metrics = compute_comprehensive_metrics(model, data, device, t_index)
    
    # Print results
    print(f"\n{'='*80}")
    print("DeepONet EVALUATION METRICS")
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
    save_dir = os.path.join(os.path.dirname(checkpoint_path), 'deeponet_evaluation')
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
    parser = argparse.ArgumentParser(description='Evaluate trained DeepONet model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat',
                       help='Path to data file')
    parser.add_argument('--t_index', type=int, default=100,
                       help='Time index for evaluation')
    parser.add_argument('--n_sensors', type=int, default=2500,
                       help='Number of sensor points')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden layer dimension')
    parser.add_argument('--depth', type=int, default=4,
                       help='Network depth')
    parser.add_argument('--p', type=int, default=100,
                       help='Inner product dimension')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluate_deeponet(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        t_index=args.t_index,
        n_sensors=args.n_sensors,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        p=args.p,
        device=args.device
    )


if __name__ == '__main__':
    main()