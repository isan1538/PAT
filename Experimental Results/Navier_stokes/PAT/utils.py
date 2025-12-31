import os
import math
import time
import argparse
import random
import numpy as np
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# Navier-Stokes Residuals
# =============================================================================

def ns_residuals_direct(uvp: torch.Tensor, xyt: torch.Tensor, nu: float):
    """
    NS residuals for direct (u,v,p) formulation.
    
    Args:
        uvp: (B,N,3) => u,v,p
        xyt: (B,N,3) => x,y,t with requires_grad=True
        nu: kinematic viscosity
    
    Returns:
        r_u, r_v, r_c: momentum-x, momentum-y, continuity residuals
    """
    u = uvp[..., 0:1]
    v = uvp[..., 1:2]
    p = uvp[..., 2:3]

    if not xyt.requires_grad:
        xyt = xyt.requires_grad_(True)

    ones_u = torch.ones_like(u)

    # First derivatives
    grads_u = torch.autograd.grad(u, xyt, grad_outputs=ones_u, 
                                  create_graph=True, retain_graph=True)[0]
    grads_v = torch.autograd.grad(v, xyt, grad_outputs=torch.ones_like(v), 
                                  create_graph=True, retain_graph=True)[0]
    grads_p = torch.autograd.grad(p, xyt, grad_outputs=torch.ones_like(p), 
                                  create_graph=True, retain_graph=True)[0]

    u_x, u_y, u_t = grads_u[..., 0:1], grads_u[..., 1:2], grads_u[..., 2:3]
    v_x, v_y, v_t = grads_v[..., 0:1], grads_v[..., 1:2], grads_v[..., 2:3]
    p_x, p_y = grads_p[..., 0:1], grads_p[..., 1:2]

    # Second derivatives (Laplacian)
    u_xx = torch.autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x), 
                              create_graph=True, retain_graph=True)[0][..., 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, grad_outputs=torch.ones_like(u_y), 
                              create_graph=True, retain_graph=True)[0][..., 1:2]
    v_xx = torch.autograd.grad(v_x, xyt, grad_outputs=torch.ones_like(v_x), 
                              create_graph=True, retain_graph=True)[0][..., 0:1]
    v_yy = torch.autograd.grad(v_y, xyt, grad_outputs=torch.ones_like(v_y), 
                              create_graph=True, retain_graph=True)[0][..., 1:2]

    # Momentum equations
    r_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    # Continuity
    r_c = u_x + v_y
    return r_u, r_v, r_c


def ns_residuals_streamfunction(psi_p: torch.Tensor, xyt: torch.Tensor, nu: float):
    """
    NS residuals for streamfunction formulation.
    
    Args:
        psi_p: (B,N,2) => psi, p
        xyt: (B,N,3) => x,y,t with requires_grad=True
        nu: kinematic viscosity
    
    Returns:
        r_u, r_v: momentum residuals (continuity automatically satisfied)
    """
    psi = psi_p[..., 0:1]
    p = psi_p[..., 1:2]

    if not xyt.requires_grad:
        xyt = xyt.requires_grad_(True)

    # Derive velocity from streamfunction: u = ∂psi/∂y, v = -∂psi/∂x
    grads_psi = torch.autograd.grad(psi, xyt, grad_outputs=torch.ones_like(psi),
                                    create_graph=True, retain_graph=True)[0]
    psi_x = grads_psi[..., 0:1]
    psi_y = grads_psi[..., 1:2]
    
    u = psi_y
    v = -psi_x

    # First derivatives of u, v
    grads_u = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
    grads_v = torch.autograd.grad(v, xyt, grad_outputs=torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
    grads_p = torch.autograd.grad(p, xyt, grad_outputs=torch.ones_like(p),
                                  create_graph=True, retain_graph=True)[0]

    u_x, u_y, u_t = grads_u[..., 0:1], grads_u[..., 1:2], grads_u[..., 2:3]
    v_x, v_y, v_t = grads_v[..., 0:1], grads_v[..., 1:2], grads_v[..., 2:3]
    p_x, p_y = grads_p[..., 0:1], grads_p[..., 1:2]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True)[0][..., 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, grad_outputs=torch.ones_like(u_y),
                              create_graph=True, retain_graph=True)[0][..., 1:2]
    v_xx = torch.autograd.grad(v_x, xyt, grad_outputs=torch.ones_like(v_x),
                              create_graph=True, retain_graph=True)[0][..., 0:1]
    v_yy = torch.autograd.grad(v_y, xyt, grad_outputs=torch.ones_like(v_y),
                              create_graph=True, retain_graph=True)[0][..., 1:2]

    # Momentum equations
    r_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return r_u, r_v, u, v


# =============================================================================
# Data Handling
# =============================================================================

class CylinderWakeData:
    """Loads and manages cylinder wake dataset."""
    def __init__(self, mat_path: str, seed: int = 0):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Could not find {mat_path}")
        
        data = scipy.io.loadmat(mat_path)
        self.U_star = data["U_star"]  # (N,2,T)
        self.p_star = data["p_star"]  # (N,T)
        self.t_star = data["t"]       # (T,1)
        self.X_star = data["X_star"]  # (N,2)

        self.N = self.X_star.shape[0]
        self.T = self.t_star.shape[0]

        # Flatten to NT points
        XX = np.tile(self.X_star[:, 0:1], (1, self.T))
        YY = np.tile(self.X_star[:, 1:2], (1, self.T))
        TT = np.tile(self.t_star, (1, self.N)).T

        UU = self.U_star[:, 0, :]
        VV = self.U_star[:, 1, :]
        PP = self.p_star

        self.x = XX.flatten()[:, None]
        self.y = YY.flatten()[:, None]
        self.t = TT.flatten()[:, None]
        self.u = UU.flatten()[:, None]
        self.v = VV.flatten()[:, None]
        self.p = PP.flatten()[:, None]

        self.NT = self.x.shape[0]
        
        # For fixed training set (like PINN)
        self.rng = np.random.RandomState(seed)

    def get_fixed_training_set(self, n: int):
        """Get fixed training set like PINN (sampled once)."""
        idx = self.rng.choice(self.NT, n, replace=False)
        xyt = np.concatenate([self.x[idx], self.y[idx], self.t[idx]], axis=1)
        uvp = np.concatenate([self.u[idx], self.v[idx], self.p[idx]], axis=1)
        return xyt, uvp

    def sample(self, n: int):
        """Random sampling (for dynamic collocation)."""
        idx = np.random.choice(self.NT, n, replace=False)
        xyt = np.concatenate([self.x[idx], self.y[idx], self.t[idx]], axis=1)
        uvp = np.concatenate([self.u[idx], self.v[idx], self.p[idx]], axis=1)
        return xyt, uvp

    def snapshot(self, t_index: int):
        """Get full spatial snapshot at specific time index."""
        t_index = int(np.clip(t_index, 0, self.T - 1))
        x = self.X_star[:, 0:1]
        y = self.X_star[:, 1:2]
        t = np.full_like(x, self.t_star[t_index, 0])
        u = self.U_star[:, 0, t_index:t_index+1]
        v = self.U_star[:, 1, t_index:t_index+1]
        p = self.p_star[:, t_index:t_index+1]
        xyt = np.concatenate([x, y, t], axis=1)
        uvp = np.concatenate([u, v, p], axis=1)
        return xyt, uvp


def plot_results(res, history, args):
    """Create comprehensive result plots."""
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    ax.semilogy(history["step"], history["loss_data"])
    ax.set_title("Total Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.semilogy(history["step"], history["data_uv"], label="data(u,v)")
    ax.semilogy(history["step"], history["data_p"], label="data(p)")
    ax.semilogy(history["step"], history["pde"], label="PDE")
    ax.semilogy(history["step"], history["cont"], label="continuity")
    ax.set_title("Loss Components")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    if len(history["eval_step"]) > 0:
        ax.semilogy(history["eval_step"], history["eval_mse"], label="MSE")
        ax.semilogy(history["eval_step"], history["eval_pde"], label="PDE")
        ax.set_title("Evaluation Metrics")
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(history["step"], history["lr"])
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(args.out_dir, f"training_curves_{args.mode}.png")
    ensure_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")
    
    # 2. Prediction visualization
    x = res["x"]
    y = res["y"]
    uvp_true = res["uvp_true"]
    uvp_pred = res["uvp_pred"]
    
    # Detect grid structure
    xu = np.unique(np.round(x, 10))
    yu = np.unique(np.round(y, 10))
    nx, ny = len(xu), len(yu)
    
    # Try to reshape to grid
    try:
        # Create mapping for structured grid
        x_to_i = {val: i for i, val in enumerate(xu)}
        y_to_j = {val: j for j, val in enumerate(yu)}
        
        def to_grid(values):
            grid = np.full((ny, nx), np.nan)
            for xi, yi, val in zip(np.round(x, 10), np.round(y, 10), values):
                grid[y_to_j[yi], x_to_i[xi]] = val
            return grid
        
        is_grid = True
    except:
        is_grid = False
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    
    for i, var in enumerate(['u', 'v', 'p']):
        true_val = uvp_true[:, i]
        pred_val = uvp_pred[:, i]
        err_val = np.abs(pred_val - true_val)
        
        if is_grid:
            # Smooth contour plots
            true_grid = to_grid(true_val)
            pred_grid = to_grid(pred_val)
            err_grid = to_grid(err_val)
            
            extent = [xu.min(), xu.max(), yu.min(), yu.max()]
            
            # True
            ax = axes[i, 0]
            im = ax.imshow(true_grid, origin='lower', aspect='auto', 
                          extent=extent, cmap='viridis', interpolation='bilinear')
            ax.set_title(f'{var} (True)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)
            
            # Predicted
            ax = axes[i, 1]
            im = ax.imshow(pred_grid, origin='lower', aspect='auto',
                          extent=extent, cmap='viridis', interpolation='bilinear')
            ax.set_title(f'{var} (Pred)')
            ax.set_xlabel('x')
            plt.colorbar(im, ax=ax)
            
            # Error
            ax = axes[i, 2]
            im = ax.imshow(err_grid, origin='lower', aspect='auto',
                          extent=extent, cmap='hot', interpolation='bilinear')
            ax.set_title(f'{var} (Abs Error)')
            ax.set_xlabel('x')
            plt.colorbar(im, ax=ax)
        else:
            # Fallback to scatter for unstructured data
            ax = axes[i, 0]
            sc = ax.scatter(x, y, c=true_val, s=3, cmap='viridis')
            ax.set_title(f'{var} (True)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(sc, ax=ax)
            
            ax = axes[i, 1]
            sc = ax.scatter(x, y, c=pred_val, s=3, cmap='viridis')
            ax.set_title(f'{var} (Pred)')
            ax.set_xlabel('x')
            plt.colorbar(sc, ax=ax)
            
            ax = axes[i, 2]
            sc = ax.scatter(x, y, c=err_val, s=3, cmap='hot')
            ax.set_title(f'{var} (Abs Error)')
            ax.set_xlabel('x')
            plt.colorbar(sc, ax=ax)
    
    plt.suptitle(f"PAT-{args.mode} | MSE={res['mse_total']:.3e} | "
                f"PDE={res['pde']:.3e} | t_index={args.eval_t_index}", 
                y=0.995)
    plt.tight_layout()
    save_path = os.path.join(args.out_dir, f"predictions_{args.mode}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved predictions to {save_path}")




def compute_additional_metrics(model, data, device, t_index, nu, mode='direct'):
    """
    Compute Rel-L2, PSNR, and PDE residual for trained model
    
    Args:
        model: Trained PATModelNS
        data: CylinderWakeData instance
        device: torch device
        t_index: time snapshot index
        nu: viscosity coefficient
        mode: 'direct', 'streamfunction', or 'pure'
    
    Returns:
        dict with rel_l2, psnr, pde_residual
    """
    model.eval()
    
    # Get full snapshot
    xyt_all, uvp_all = data.get_snapshot(t_index)
    N = xyt_all.shape[0]
    
    ctx_pos = to_device(xyt_all, device).unsqueeze(0)
    ctx_feats = to_device(uvp_all[:, 0:2], device).unsqueeze(0)
    
    # Predict in batches
    batch_query = 8192
    outs = []
    for i in range(0, N, batch_query):
        xyt_q = ctx_pos[:, i:i+batch_query, :]
        with torch.no_grad():
            out_i = model(ctx_feats, ctx_pos, xyt_q)
        outs.append(out_i)
    pred = torch.cat(outs, dim=1).squeeze(0)
    
    # Handle streamfunction mode
    if mode == 'streamfunction':
        with torch.enable_grad():
            xyt_rg = ctx_pos.clone().requires_grad_(True)
            psi_p_out = model(ctx_feats, ctx_pos, xyt_rg)
            psi = psi_p_out[..., 0:1]
            
            grads_psi = torch.autograd.grad(psi, xyt_rg,
                                           grad_outputs=torch.ones_like(psi),
                                           create_graph=True)[0]
            u_pred = grads_psi[..., 1:2].squeeze(0)
            v_pred = -grads_psi[..., 0:1].squeeze(0)
        p_pred = pred[:, 1:2]
        uvp_pred = torch.cat([u_pred, v_pred, p_pred], dim=1)
    else:
        uvp_pred = pred
    
    uvp_true = to_device(uvp_all, device)
    
    # Relative L2 norm
    def rel_l2(pred, true):
        return (torch.norm(pred - true) / torch.norm(true)).item()
    
    rel_l2_u = rel_l2(uvp_pred[:, 0], uvp_true[:, 0])
    rel_l2_v = rel_l2(uvp_pred[:, 1], uvp_true[:, 1])
    rel_l2_p = rel_l2(uvp_pred[:, 2], uvp_true[:, 2])
    rel_l2_avg = (rel_l2_u + rel_l2_v + rel_l2_p) / 3.0
    
    # PSNR
    def psnr(pred, true):
        mse = F.mse_loss(pred, true)
        if mse == 0:
            return float('inf')
        data_range = true.max() - true.min()
        return 20 * torch.log10(data_range / torch.sqrt(mse)).item()
    
    psnr_u = psnr(uvp_pred[:, 0], uvp_true[:, 0])
    psnr_v = psnr(uvp_pred[:, 1], uvp_true[:, 1])
    psnr_p = psnr(uvp_pred[:, 2], uvp_true[:, 2])
    psnr_avg = (psnr_u + psnr_v + psnr_p) / 3.0
    
    # PDE residual
    with torch.enable_grad():
        xyt_rg = ctx_pos.clone().requires_grad_(True)
        
        if mode == 'streamfunction':
            psi_p = model(ctx_feats, ctx_pos, xyt_rg)
            r_u, r_v, _, _ = ns_residuals_streamfunction(psi_p, xyt_rg, nu)
            pde_res = (r_u.square().mean() + r_v.square().mean()).item()
        else:
            uvp_q = model(ctx_feats, ctx_pos, xyt_rg)
            r_u, r_v, r_c = ns_residuals_direct(uvp_q, xyt_rg, nu)
            pde_res = (r_u.square().mean() + r_v.square().mean() + r_c.square().mean()).item()
    
    return {
        'rel_l2_u': rel_l2_u,
        'rel_l2_v': rel_l2_v,
        'rel_l2_p': rel_l2_p,
        'rel_l2_avg': rel_l2_avg,
        'psnr_u': psnr_u,
        'psnr_v': psnr_v,
        'psnr_p': psnr_p,
        'psnr_avg': psnr_avg,
        'pde_residual': pde_res,
    }