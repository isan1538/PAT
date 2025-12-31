#!/usr/bin/env python3
"""

comparing metrics calculation (PSNR, Rel_L2, ...) added in the main function, added to 1227.py


Run:
###############################################################
Method: Pure data-driven transformer without physics constraints
################################################################
python pat_train_navier_stokes_1228.py --mode pure --eval_t_index 100 --n_train 2500 \
  --steps 5000 --save_path checkpoints/pat_pure.pt \
  --out_dir checkpoints/pat_pure  

################################################################
  Method: Physics-Aware Transformer with PDE constraints and diffusion bias
################################################################
python pat_train_navier_stokes_1228.py --mode physics --eval_t_index 100 --n_train 2500 \
  --steps 5000 --save_path checkpoints/pat_physics.pt \
  --out_dir checkpoints/pat_physics


################################################################
  Method: PAT with streamfunction formulation (like PINN)
################################################################
python pat_train_navier_stokes_1228.py --mode streamfunction --eval_t_index 100 --n_train 2500 \
  --steps 5000 --save_path checkpoints/pat_streamfunction.pt \
  --out_dir checkpoints/pat_streamfunction
  
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

from utils import *

from pat_model_1208 import (
    PATConfig, LayerNorm, PhysSelfAttention, PATBlock, 
    CrossAttention, SIRENLayer, FiLMHyper
)


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
        "step": [], "loss": [], "loss_data": [], "data_uv": [], "data_p": [], 
        "pde": [], "cont": [], "lr": [],
        "eval_step": [], "eval_mse": [], "eval_pde": []
    }
    
    best_eval = float("inf")
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
        ################## 1227 ###################
        lossdata = args.w_data * loss_data_uv
        ################## 1227 ###################

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        sched.step()
        
        # Logging
        if step % args.print_every == 0 or step == 1:
            lr_now = sched.get_last_lr()[0]
            print(f"[{step:06d}] Data loss={lossdata.item():.3e} | " 
                  f"p={loss_p.item():.3e} | pde={loss_pde.item():.3e} | "
                  f"cont={loss_cont.item():.3e} | lr={lr_now:.2e}")
            
            history["step"].append(step)
            history["loss_data"].append(lossdata.item())
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
            print(f"         EVAL t={args.eval_t_index} | "
                  f"MSE={res['mse_total']:.3e} "
                  f"(u={res['mse_u']:.3e}, v={res['mse_v']:.3e}, "
                  f"p={res['mse_p']:.3e}) | PDE={res['pde']:.3e}")
            
            history["eval_step"].append(step)
            history["eval_mse"].append(res["mse_total"])
            history["eval_pde"].append(res["pde"])
            
            if res["mse_total"] < best_eval:
                best_eval = res["mse_total"]
                if args.save_path:
                    ensure_dir(args.save_path)
                    torch.save({
                        "model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "step": step,
                        "eval_mse": best_eval,
                        "args": vars(args),
                    }, args.save_path.replace(".pt", "_best.pt"))
                    print(f"         ✓ Saved best checkpoint")
        
        # Periodic checkpoints
        if args.save_path and step % args.save_every == 0:
            ensure_dir(args.save_path)
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "step": step,
                "history": history,
                "args": vars(args),
            }, args.save_path)
    
    elapsed = time.time() - t0
    print(f"\nDone! Best MSE={best_eval:.3e}, Time={elapsed/60:.1f} min")
    
    # Save final checkpoint
    if args.save_path:
        ensure_dir(args.save_path)
        torch.save({
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "step": args.steps,
            "history": history,
            "args": vars(args),
        }, args.save_path.replace(".pt", "_final.pt"))
        print(f"Saved final checkpoint")
    
    # Final evaluation and plot
    res = eval_full_snapshot(model, data, device, args.eval_t_index, 
                            nu, mode=args.mode)
    if args.plot:
        plot_results(res, history, args)

    return model, history



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
    ################## 1227 ###################
    ap.add_argument("--w_data", type=float, default=0.1)
    ################## 1227 ###################
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
    ap.add_argument("--print_every", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=500)
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
    
    # Train the model
    model, history = train(args)
    
    # After training, compute final metrics
    print("\nComputing additional metrics...")
    
    # Recreate necessary variables
    device = torch.device(args.device)
    data = CylinderWakeData(args.data_path, seed=args.seed)
    nu = 1.0 / args.Re
    mode = args.mode
    
    # Compute comprehensive metrics
    metrics = compute_additional_metrics(model, data, device, args.eval_t_index, nu, mode=mode)
    
    print(f"\n{'='*80}")
    print("FINAL EVALUATION METRICS")
    print(f"{'='*80}")
    print(f"Rel-L2 (u):      {metrics['rel_l2_u']:.6f}")
    print(f"Rel-L2 (v):      {metrics['rel_l2_v']:.6f}")
    print(f"Rel-L2 (p):      {metrics['rel_l2_p']:.6f}")
    print(f"Rel-L2 (avg):    {metrics['rel_l2_avg']:.6f}")
    print(f"\nPSNR (u):        {metrics['psnr_u']:.2f} dB")
    print(f"PSNR (v):        {metrics['psnr_v']:.2f} dB")
    print(f"PSNR (p):        {metrics['psnr_p']:.2f} dB")
    print(f"PSNR (avg):      {metrics['psnr_avg']:.2f} dB")
    print(f"\nPDE Residual:    {metrics['pde_residual']:.6e}")
    print(f"{'='*80}\n")
    
    # Save metrics to best checkpoint
    if args.save_path:
        best_checkpoint_path = args.save_path.replace(".pt", "_best.pt")
        checkpoint = torch.load(best_checkpoint_path)
        checkpoint['metrics'] = metrics
        torch.save(checkpoint, best_checkpoint_path)
        print(f"Saved metrics to checkpoint: {best_checkpoint_path}")
