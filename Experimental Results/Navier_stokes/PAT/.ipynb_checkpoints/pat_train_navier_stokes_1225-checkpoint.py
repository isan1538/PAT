#!/usr/bin/env python3
"""
Adding metrics calculation to 1214.py

python pat_train_navier_stokes_1225.py \
    --mode physics \
    --data_path ./cylinder_nektar_wake.mat \
    --device cuda \
    --steps 500 \
    --save_path checkpoints/pat_physics.pt \
    --out_dir outputs/pat_physics
  
"""


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

from pat_model_1208 import (
    PATConfig, LayerNorm, PhysSelfAttention, PATBlock, 
    CrossAttention, SIRENLayer, FiLMHyper
)


# =============================================================================
# Utilities
# =============================================================================

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
# Multi-output FiLM-SIREN
# =============================================================================

class FiLMSIRENMulti(nn.Module):
    """FiLM-modulated SIREN with configurable output dimension."""
    def __init__(
        self,
        in_dim: int,
        width: int,
        depth: int,
        omega0: float,
        hyper_in_dim: int,
        hyper_hidden: int,
        out_dim: int = 3,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.omega0 = omega0
        self.hyper = FiLMHyper(hyper_in_dim, hyper_hidden, depth, width)

        layers = []
        layers.append(SIRENLayer(in_dim, width, omega0=omega0, is_first=True))
        for _ in range(depth - 2):
            layers.append(SIRENLayer(width, width, omega0=omega0, is_first=False))
        self.layers = nn.ModuleList(layers)

        self.final = nn.Linear(width, out_dim)
        with torch.no_grad():
            self.final.weight.uniform_(
                -math.sqrt(6 / width) / omega0, 
                math.sqrt(6 / width) / omega0
            )

    def forward(self, x, g, cglob):
        B, Nq, _ = x.shape
        g_in = torch.cat([g, cglob.expand(B, Nq, -1)], dim=-1)
        gammas, betas, omegas = self.hyper(g_in)

        h = self.layers[0](x)
        for i, layer in enumerate(self.layers[1:], start=1):
            gamma = gammas[i]
            beta = betas[i]
            omega = omegas[i]
            h = gamma * h + beta
            h = torch.sin(omega * layer.linear(h))
        out = self.final(h)
        return out


# =============================================================================
# PAT Model for Navier-Stokes
# =============================================================================

class PATModelNS(nn.Module):
    """
    PAT for 2D Navier-Stokes with three formulation modes:
    
    1. 'direct': Predicts (u,v,p) directly, enforces continuity via loss
    2. 'streamfunction': Predicts (psi,p), derives u,v (continuity by construction)
    3. 'pure': Pure transformer (no physics bias)
    """
    def __init__(self, cfg: PATConfig, out_dim: int = 3, mode: str = 'direct'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode  # 'direct', 'streamfunction', or 'pure'
        
        # If streamfunction mode, output is (psi, p) instead of (u,v,p)
        if mode == 'streamfunction':
            out_dim = 2

        self.patch_embed = nn.Linear(cfg.d_patch, cfg.n_embd, bias=cfg.bias)
        self.pos_enc = nn.Linear(cfg.d_pos, cfg.n_embd, bias=cfg.bias)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.n_embd) * 0.02)

        self.blocks = nn.ModuleList([
            PATBlock(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.bias, 
                    cfg.use_gradient_checkpointing)
            for _ in range(cfg.n_layer)
        ])
        self.ln_ctx = LayerNorm(cfg.n_embd, bias=cfg.bias)

        query_dim = 128
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_pos, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )

        self.cross = CrossAttention(
            q_dim=query_dim,
            ctx_dim=cfg.n_embd,
            out_dim=cfg.n_embd,
            n_head=cfg.n_head,
            bias=cfg.bias,
            dropout=cfg.dropout,
        )

        self.inr = FiLMSIRENMulti(
            in_dim=query_dim,
            width=128,
            depth=4,
            omega0=30.0,
            hyper_in_dim=cfg.n_embd + cfg.n_embd,
            hyper_hidden=256,
            out_dim=out_dim,
        )

        self.register_buffer("neg_inf", torch.tensor(float("-inf")))
        self.register_buffer("neg_inf_value", torch.tensor(-1e9))

    def diffusion_bias_2d(self, ctx_pos: torch.Tensor) -> torch.Tensor:
        """
        Diffusion-like physics bias in 2D for Navier-Stokes.
        Only used if cfg.alpha > 0 and mode != 'pure'
        """
        if self.mode == 'pure' or self.cfg.alpha == 0.0:
            # Return zero bias for pure transformer mode
            B, P, _ = ctx_pos.shape
            return torch.zeros(B, 1, P, P, device=ctx_pos.device)
        
        B, P, _ = ctx_pos.shape
        x = ctx_pos[..., 0].unsqueeze(-1)
        y = ctx_pos[..., 1].unsqueeze(-1)
        t = ctx_pos[..., 2].unsqueeze(-1)

        dx2 = (x - x.transpose(1, 2)) ** 2
        dy2 = (y - y.transpose(1, 2)) ** 2
        r2 = dx2 + dy2

        dt = t - t.transpose(1, 2)
        mask = dt > 0

        safe_dt = torch.clamp(dt, min=1e-6)
        safe_r2 = torch.clamp(r2, min=1e-10)

        nu = torch.clamp(torch.tensor(self.cfg.nu_bar, device=ctx_pos.device), min=1e-8)

        log_term = -torch.log(torch.clamp(4 * math.pi * nu * safe_dt, min=1e-10))
        exp_term = -safe_r2 / torch.clamp(4 * nu * safe_dt, min=1e-10)
        exp_term = torch.clamp(exp_term, min=-50, max=50)

        logG = log_term + exp_term
        logG = torch.clamp(logG, min=-50, max=50)

        logG = torch.where(mask, self.cfg.alpha * logG, self.neg_inf)
        return logG.unsqueeze(1)

    def encode_context(self, ctx_feats: torch.Tensor, ctx_pos: torch.Tensor):
        B, P, _ = ctx_feats.shape

        e = self.patch_embed(ctx_feats) + self.pos_enc(ctx_pos)
        cls = self.cls.expand(B, -1, -1)
        C = torch.cat([cls, e], dim=1)

        gamma = self.diffusion_bias_2d(ctx_pos)
        gamma_full = gamma.new_zeros(B, 1, P + 1, P + 1)
        gamma_full[:, :, 1:, 1:] = gamma

        for blk in self.blocks:
            C = blk(C, gamma_bias=gamma_full)

        C = self.ln_ctx(C)
        cglob = C[:, :1, :]
        C_ctx = C[:, 1:, :]
        return C_ctx, cglob

    def forward(self, ctx_feats: torch.Tensor, ctx_pos: torch.Tensor, 
                xyt_q: torch.Tensor):
        """
        Returns:
            - If mode='direct': (u, v, p) with shape (B, Nq, 3)
            - If mode='streamfunction': (psi, p) with shape (B, Nq, 2)
        """
        C, cglob = self.encode_context(ctx_feats, ctx_pos)
        phi = self.ff(xyt_q)
        g = self.cross(phi, C)
        out = self.inr(phi, g, cglob)
        return out


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


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def eval_full_snapshot(model, data: CylinderWakeData, device, t_index: int, 
                       nu: float, batch_query: int = 8192, mode: str = 'direct'):
    """Evaluate on full spatial snapshot at given time index."""
    model.eval()
    
    # Get full snapshot
    xyt_all, uvp_all = data.snapshot(t_index)
    N = xyt_all.shape[0]
    
    # Use same data as context (simulating perfect sparse observations)
    # For fair comparison with PINN, we provide all training data as context
    ctx_pos = to_device(xyt_all, device).unsqueeze(0)  # (1,N,3)
    ctx_feats = to_device(uvp_all[:, 0:2], device).unsqueeze(0)  # (1,N,2)
    
    # Query all points
    xyt_q = to_device(xyt_all, device).unsqueeze(0)
    
    # Predict in batches
    outs = []
    for i in range(0, xyt_q.shape[1], batch_query):
        out_i = model(ctx_feats, ctx_pos, xyt_q[:, i:i+batch_query, :])
        outs.append(out_i)
    pred = torch.cat(outs, dim=1).squeeze(0)  # (N, 2 or 3)
    
    # Extract u, v, p depending on mode
    if mode == 'streamfunction':
        # Compute u, v from streamfunction (need gradients)
        with torch.enable_grad():
            xyt_rg = xyt_q.clone().requires_grad_(True)
            psi_p_out = model(ctx_feats, ctx_pos, xyt_rg)
            psi = psi_p_out[..., 0:1]
            
            grads_psi = torch.autograd.grad(psi, xyt_rg, 
                                           grad_outputs=torch.ones_like(psi),
                                           create_graph=True)[0]
            u_pred = grads_psi[..., 1:2].squeeze(0)  # ∂psi/∂y
            v_pred = -grads_psi[..., 0:1].squeeze(0)  # -∂psi/∂x
        p_pred = pred[:, 1:2]
        uvp_pred = torch.cat([u_pred, v_pred, p_pred], dim=1)
    else:
        uvp_pred = pred  # Already (u,v,p)
    
    uvp_true = to_device(uvp_all, device)
    
    # Compute MSE
    mse_u = F.mse_loss(uvp_pred[:, 0], uvp_true[:, 0]).item()
    mse_v = F.mse_loss(uvp_pred[:, 1], uvp_true[:, 1]).item()
    mse_p = F.mse_loss(uvp_pred[:, 2], uvp_true[:, 2]).item()
    mse_total = (mse_u + mse_v + mse_p) / 3.0
    
    # Compute PDE residuals (need gradients enabled)
    with torch.enable_grad():
        xyt_rg = xyt_q.clone().requires_grad_(True)
        
        if mode == 'streamfunction':
            psi_p = model(ctx_feats, ctx_pos, xyt_rg)
            r_u, r_v, _, _ = ns_residuals_streamfunction(psi_p, xyt_rg, nu)
            pde = (r_u.square().mean() + r_v.square().mean()).item()
        else:
            uvp_q = model(ctx_feats, ctx_pos, xyt_rg)
            r_u, r_v, r_c = ns_residuals_direct(uvp_q, xyt_rg, nu)
            pde = (r_u.square().mean() + r_v.square().mean() + r_c.square().mean()).item()
    
    return {
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_p": mse_p,
        "mse_total": mse_total,
        "pde": pde,
        "uvp_pred": uvp_pred.detach().cpu().numpy(),
        "uvp_true": uvp_all,
        "x": xyt_all[:, 0],
        "y": xyt_all[:, 1],
    }

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
    xyt_all, uvp_all = data.snapshot(t_index)
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
    
# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device(args.device)
    set_seed(args.seed)
    
    data = CylinderWakeData(args.data_path, seed=args.seed)
    nu = 1.0 / args.Re
    
    # Get fixed training set (like PINN)
    xyt_train, uvp_train = data.get_fixed_training_set(args.n_train)
    xyt_train = to_device(xyt_train, device).requires_grad_(True)
    uvp_train = to_device(uvp_train, device)
    
    # PAT config
    cfg = PATConfig()
    cfg.d_patch = 2  # (u,v) as features
    cfg.d_pos = 3    # (x,y,t)
    cfg.n_embd = args.n_embd
    cfg.n_head = args.n_head
    cfg.n_layer = args.n_layer
    cfg.dropout = args.dropout
    cfg.use_gradient_checkpointing = args.use_checkpointing
    cfg.nu_bar = nu
    
    # Set alpha based on mode
    if args.mode == 'pure':
        cfg.alpha = 0.0
    elif args.mode == 'physics':
        cfg.alpha = args.alpha
    else:  # streamfunction
        cfg.alpha = args.alpha
    
    model = PATModelNS(cfg, out_dim=3, mode=args.mode).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*80}")
    print("PAT for 2D Navier-Stokes - Improved for PINN Comparison")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Data: {args.data_path} | N={data.N} | T={data.T}")
    print(f"Re={args.Re} -> nu={nu:.6f}")
    print(f"Training points: {args.n_train} (fixed like PINN)")
    print(f"Collocation points: {args.n_colloc}")
    print(f"Alpha (physics bias): {cfg.alpha}")
    print(f"Loss weights: w_data={args.w_data}, w_p={args.w_p}, " 
          f"w_pde={args.w_pde}, w_cont={args.w_cont}")
    print(f"Steps: {args.steps}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Model parameters: {n_params:,}")
    print(f"Eval at t_index={args.eval_t_index}")
    print(f"{'='*80}\n")
    
    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                             weight_decay=args.weight_decay)
    
    # LR scheduler with warmup + cosine decay
    def lr_lambda(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        prog = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    
    # Training history
    history = {
        "step": [], "loss": [], "data_uv": [], "data_p": [], 
        "pde": [], "cont": [], "lr": [],
        "eval_step": [], "eval_mse": [], "eval_pde": [],
        "eval_psnr_u": [], "eval_psnr_v": [], "eval_psnr_p": [], "eval_psnr_avg": []
    }
    
    best_eval = float("inf")
    best_loss = float("inf")
    t0 = time.time()
    
    for step in range(1, args.steps + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        
        # Build context from training data (can subsample if needed)
        if args.batch_size == 1:
            # Use all training data as context
            ctx_pos = xyt_train.unsqueeze(0)  # (1,N,3)
            ctx_feats = uvp_train[:, 0:2].unsqueeze(0)  # (1,N,2)
            obs_uvp = uvp_train.unsqueeze(0)  # (1,N,3)
        else:
            # For batch_size > 1, could subsample differently per batch element
            # For now, keep it simple
            ctx_pos = xyt_train.unsqueeze(0).expand(args.batch_size, -1, -1)
            ctx_feats = uvp_train[:, 0:2].unsqueeze(0).expand(args.batch_size, -1, -1)
            obs_uvp = uvp_train.unsqueeze(0).expand(args.batch_size, -1, -1)
        
        # Sample collocation points
        xyt_col_list = []
        for _ in range(args.batch_size):
            xyt_c, _ = data.sample(args.n_colloc)
            xyt_col_list.append(to_device(xyt_c, device))
        xyt_col = torch.stack(xyt_col_list, dim=0).requires_grad_(True)
        
        # Data loss on training points
        if args.mode == 'streamfunction':
            # For streamfunction, we can't directly supervise u,v
            # Instead, supervise on pressure and use PDE loss for velocity
            pred = model(ctx_feats, ctx_pos, ctx_pos)
            loss_p = F.mse_loss(pred[..., 1:2], obs_uvp[..., 2:3])
            loss_data_uv = torch.tensor(0.0, device=device)
        else:
            pred = model(ctx_feats, ctx_pos, ctx_pos)
            loss_u = F.mse_loss(pred[..., 0:1], obs_uvp[..., 0:1])
            loss_v = F.mse_loss(pred[..., 1:2], obs_uvp[..., 1:2])
            loss_p = F.mse_loss(pred[..., 2:3], obs_uvp[..., 2:3])
            loss_data_uv = loss_u + loss_v
        
        # PDE residuals at collocation points
        pred_col = model(ctx_feats, ctx_pos, xyt_col)
        
        if args.mode == 'streamfunction':
            r_u, r_v, _, _ = ns_residuals_streamfunction(pred_col, xyt_col, nu)
            loss_pde = r_u.square().mean() + r_v.square().mean()
            loss_cont = torch.tensor(0.0, device=device)  # Continuity automatic
        else:
            r_u, r_v, r_c = ns_residuals_direct(pred_col, xyt_col, nu)
            loss_pde = r_u.square().mean() + r_v.square().mean()
            loss_cont = r_c.square().mean()
        
        # Total loss
        loss = (args.w_data * loss_data_uv + 
                args.w_p * loss_p + 
                args.w_pde * loss_pde + 
                args.w_cont * loss_cont)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        sched.step()

        
        
        # Logging
        if step % args.print_every == 0 or step == 1:
            lr_now = sched.get_last_lr()[0]
            print(f"[{step:06d}] loss={loss.item():.3e} | " 
                  f"data(u,v)={loss_data_uv.item():.3e} | "
                  f"p={loss_p.item():.3e} | pde={loss_pde.item():.3e} | "
                  f"cont={loss_cont.item():.3e} | lr={lr_now:.2e}")
            
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["data_uv"].append(loss_data_uv.item())
            history["data_p"].append(loss_p.item())
            history["pde"].append(loss_pde.item())
            history["cont"].append(loss_cont.item())
            history["lr"].append(lr_now)
        
        # Evaluation
        if step % args.eval_every == 0 or step == args.steps:
            res = eval_full_snapshot(model, data, device, args.eval_t_index, 
                                    nu, mode=args.mode)
            
            # Compute PSNR for each variable at eval_t_index
            # Get predictions and ground truth
            xyt_all, uvp_all = data.snapshot(args.eval_t_index)
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
            if args.mode == 'streamfunction':
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
            
            # PSNR calculation
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
            
            psnr_metrics = {
                'psnr_u_t100': psnr_u,
                'psnr_v_t100': psnr_v,
                'psnr_p_t100': psnr_p,
                'psnr_avg_t100': psnr_avg,
            }
            
            print(f"         EVAL t={args.eval_t_index} | "
                  f"MSE={res['mse_total']:.3e} "
                  f"(u={res['mse_u']:.3e}, v={res['mse_v']:.3e}, "
                  f"p={res['mse_p']:.3e}) | PDE={res['pde']:.3e}")
            
            print(f"         PSNR t={args.eval_t_index} | "
                  f"U={psnr_u:.2f} dB, "
                  f"V={psnr_v:.2f} dB, "
                  f"P={psnr_p:.2f} dB, "
                  f"Avg={psnr_avg:.2f} dB")
            
            history["eval_step"].append(step)
            history["eval_mse"].append(res["mse_total"])
            history["eval_pde"].append(res["pde"])
            history["eval_psnr_u"].append(psnr_u)
            history["eval_psnr_v"].append(psnr_v)
            history["eval_psnr_p"].append(psnr_p)
            history["eval_psnr_avg"].append(psnr_avg)
            
            if res["mse_total"] < best_eval:
                best_eval = res["mse_total"]
                if args.save_path:
                    ensure_dir(args.save_path)
                    checkpoint_data = {
                        "model_state_dict": model.state_dict(),
                        "config": cfg.__dict__,
                        "step": step,
                        "best_mse": best_eval,
                        "args": vars(args),
                        "mode": args.mode,
                    }
                    # Add PSNR metrics
                    checkpoint_data.update(psnr_metrics)
                    
                    torch.save(checkpoint_data, args.save_path.replace(".pt", "_best.pt"))
                    print(f"         ✓ Saved best checkpoint with PSNR metrics")
        
        best_loss = loss
        
        # Periodic checkpoints
        if args.save_path and step % args.save_every == 0:
            ensure_dir(args.save_path)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "step": step,
                "history": history,
                "args": vars(args),
                "mode": args.mode,
            }, args.save_path)
    
    elapsed = time.time() - t0
    print(f"\nDone! Best MSE={best_eval:.3e}, Best mean PSNR={psnr_avg}, Time={elapsed/60:.1f} min")
    
    # Save final checkpoint
    if args.save_path:
        ensure_dir(args.save_path)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "step": args.steps,
            "history": history,
            "args": vars(args),
            "mode": args.mode,
        }, args.save_path.replace(".pt", "_final.pt"))
        print(f"Saved final checkpoint")
    
    # Final evaluation and plot
    res = eval_full_snapshot(model, data, device, args.eval_t_index, 
                            nu, mode=args.mode)


    if args.plot:
        plot_results(res, history, args)


def plot_results(res, history, args):
    """Create comprehensive result plots."""
    # 1. Training curves (now with PSNR)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    ax.semilogy(history["step"], history["loss"])
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
    
    # NEW: PSNR curves
    ax = axes[2, 0]
    if len(history["eval_step"]) > 0:
        ax.plot(history["eval_step"], history["eval_psnr_u"], 'o-', label="PSNR(u)", linewidth=2, markersize=4)
        ax.plot(history["eval_step"], history["eval_psnr_v"], 's-', label="PSNR(v)", linewidth=2, markersize=4)
        ax.plot(history["eval_step"], history["eval_psnr_p"], '^-', label="PSNR(p)", linewidth=2, markersize=4)
        ax.set_title(f"PSNR Evolution (t={args.eval_t_index})")
        ax.set_xlabel("Step")
        ax.set_ylabel("PSNR (dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    if len(history["eval_step"]) > 0:
        ax.plot(history["eval_step"], history["eval_psnr_avg"], 'o-', 
               label="PSNR(avg)", linewidth=2, markersize=4, color='black')
        ax.set_title(f"Average PSNR (t={args.eval_t_index})")
        ax.set_xlabel("Step")
        ax.set_ylabel("PSNR (dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal reference lines
        ax.axhline(y=40, color='g', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (40 dB)')
        ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Good (30 dB)')
        ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Fair (20 dB)')
        ax.legend()
    
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



def parse_args():
    ap = argparse.ArgumentParser()
    
    # Data
    ap.add_argument("--data_path", type=str, default="./cylinder_nektar_wake.mat")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    
    # Physics
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=1.0, 
                   help="Physics bias strength")
    
    # Mode selection
    ap.add_argument("--mode", type=str, default="physics", 
                   choices=["physics", "pure", "streamfunction"],
                   help="physics: PAT with PDE loss and bias | "
                        "pure: transformer only | "
                        "streamfunction: continuity by construction")
    
    # Training data
    ap.add_argument("--n_train", type=int, default=2500,
                   help="Number of fixed training points (like PINN)")
    ap.add_argument("--n_colloc", type=int, default=5000,
                   help="Number of collocation points per iteration")
    
    # Loss weights
    ap.add_argument("--w_data", type=float, default=1.0)
    ap.add_argument("--w_p", type=float, default=1.0,
                   help="Pressure supervision (1.0 to match PINN)")
    ap.add_argument("--w_pde", type=float, default=1.0)
    ap.add_argument("--w_cont", type=float, default=1.0)
    
    # Model
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_checkpointing", action="store_true")
    
    # Training
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    
    # Logging
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_t_index", type=int, default=100,
                   help="Time index for evaluation (100 to match PINN)")
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--save_path", type=str, default="checkpoints/pat_ns_improved.pt")
    ap.add_argument("--out_dir", type=str, default="outputs")
    
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.save_path)
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)