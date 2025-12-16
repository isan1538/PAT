"""
PAT Training for 2D Heat Diffusion Equation

2D Heat Equation: ∂u/∂t = ν(∂²u/∂x² + ∂²u/∂y²)
Domain: (x,y,t) ∈ [0,1] × [0,1] × [0,1]
Boundary Conditions: u(0,y,t) = u(1,y,t) = u(x,0,t) = u(x,1,t) = 0

Analytical Solution (for verification):
u(x,y,t) = exp(-ν*π²*(n²+m²)*t) * sin(n*π*x) * sin(m*π*y)

Example usage:

Pure transformer (no physics):

python 1211_pat_training_2d.py \
    --M 100 \
    --nu 0.1 \
    --train_modes "1,1" "1,2" \
    --test_modes "2,2" "2,3" \
    --Nc 0 \
    --Nbc 0 \
    --Nic 0 \
    --steps 30000 \
    --batch_size 4 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --warmup_steps 1000 \
    --weight_pde 0.0 \
    --weight_bc 0.0 \
    --weight_ic 0.0 \
    --n_layers 6 \
    --n_heads 8 \
    --n_embd 256 \
    --dropout 0.1 \
    --alpha 0.0 \
    --print_every 100 \
    --eval_every 500 \
    --plot

python 1211_pat_training_2d.py --alpha 0.0 --weight_pde 0.0 --weight_bc 0.0 --weight_ic 0.0 \
      --train_modes "1,1" "1,2" --test_modes "2,2" "2,3" --M 50 --steps 5000

Physics-informed transformer:
python 1211_pat_training_2d.py \
    --M 100 \
    --nu 0.1 \
    --train_modes "1,1" "1,2" \
    --test_modes "2,2" "2,3" \
    --Nc 4096 \
    --Nbc 512 \
    --Nic 1024 \
    --steps 30000 \
    --batch_size 4 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --warmup_steps 1000 \
    --weight_pde 0.1 \
    --weight_bc 0.1 \
    --weight_ic 0.1 \
    --n_layers 6 \
    --n_heads 8 \
    --n_embd 256 \
    --dropout 0.1 \
    --alpha 1.0 \
    --print_every 100 \
    --eval_every 500 \
    --plot
"""

import math
import random
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pat_model_2d_1212 import PATConfig, PATModel


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def exact_solution_2d(x, y, t, nu, n=1, m=1):
    """
    Analytical solution for 2D heat equation with separable variables.
    
    Args:
        x, y: spatial coordinates (tensors)
        t: time coordinate (tensor)
        nu: diffusivity
        n, m: mode numbers in x and y directions
    
    Returns:
        u(x,y,t) = exp(-ν*π²*(n²+m²)*t) * sin(n*π*x) * sin(m*π*y)
    """
    factor = nu * (math.pi ** 2) * (n**2 + m**2)
    return torch.exp(-factor * t) * torch.sin(n * math.pi * x) * torch.sin(m * math.pi * y)


class SparseHeat2DDataset(Dataset):
    """
    Dataset for 2D heat equation sparse reconstruction.
    Each sample contains:
    - Sparse observations in (x,y,t) domain
    - Collocation points for PDE residual
    - Boundary condition points (4 edges)
    - Initial condition points
    """
    
    def __init__(
        self,
        num_instances,
        M_sparse,
        nu,
        modes,  # List of tuples [(n1,m1), (n2,m2), ...]
        Nc,
        Nbc,
        Nic,
        device="cuda",
    ):
        super().__init__()
        self.num_instances = num_instances
        self.M = M_sparse
        self.nu = nu
        self.modes = modes  # List of (n, m) tuples
        self.Nc = Nc
        self.Nbc = Nbc
        self.Nic = Nic
        self.device = device

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        # Randomly select a mode (n, m)
        n, m = random.choice(self.modes)
        
        # Sparse observations in (x, y, t) domain
        x_sparse = torch.rand(self.M, 1, device=self.device)
        y_sparse = torch.rand(self.M, 1, device=self.device)
        t_sparse = torch.rand(self.M, 1, device=self.device)
        u_sparse = exact_solution_2d(x_sparse, y_sparse, t_sparse, self.nu, n=n, m=m)
        xyt_sparse = torch.cat([x_sparse, y_sparse, t_sparse], dim=-1)
        
        # Collocation points for PDE residual
        x_colloc = torch.rand(self.Nc, 1, device=self.device)
        y_colloc = torch.rand(self.Nc, 1, device=self.device)
        t_colloc = torch.rand(self.Nc, 1, device=self.device)
        colloc = torch.cat([x_colloc, y_colloc, t_colloc], dim=-1)
        
        # Boundary conditions (4 edges)
        # Edge 1: x=0, y∈[0,1], t∈[0,1]
        y_bc1 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc1 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc1 = exact_solution_2d(torch.zeros_like(y_bc1), y_bc1, t_bc1, self.nu, n=n, m=m)
        xyt_bc1 = torch.cat([torch.zeros_like(y_bc1), y_bc1, t_bc1], dim=-1)
        
        # Edge 2: x=1, y∈[0,1], t∈[0,1]
        y_bc2 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc2 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc2 = exact_solution_2d(torch.ones_like(y_bc2), y_bc2, t_bc2, self.nu, n=n, m=m)
        xyt_bc2 = torch.cat([torch.ones_like(y_bc2), y_bc2, t_bc2], dim=-1)
        
        # Edge 3: x∈[0,1], y=0, t∈[0,1]
        x_bc3 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc3 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc3 = exact_solution_2d(x_bc3, torch.zeros_like(x_bc3), t_bc3, self.nu, n=n, m=m)
        xyt_bc3 = torch.cat([x_bc3, torch.zeros_like(x_bc3), t_bc3], dim=-1)
        
        # Edge 4: x∈[0,1], y=1, t∈[0,1]
        x_bc4 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc4 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc4 = exact_solution_2d(x_bc4, torch.ones_like(x_bc4), t_bc4, self.nu, n=n, m=m)
        xyt_bc4 = torch.cat([x_bc4, torch.ones_like(x_bc4), t_bc4], dim=-1)
        
        # Initial condition: t=0, (x,y)∈[0,1]²
        x_ic = torch.rand(self.Nic, 1, device=self.device)
        y_ic = torch.rand(self.Nic, 1, device=self.device)
        u_ic = exact_solution_2d(x_ic, y_ic, torch.zeros_like(x_ic), self.nu, n=n, m=m)
        xyt_ic = torch.cat([x_ic, y_ic, torch.zeros_like(x_ic)], dim=-1)
        
        return {
            "xyt_sparse": xyt_sparse,
            "u_sparse": u_sparse,
            "colloc": colloc,
            "xyt_bc1": xyt_bc1,
            "u_bc1": u_bc1,
            "xyt_bc2": xyt_bc2,
            "u_bc2": u_bc2,
            "xyt_bc3": xyt_bc3,
            "u_bc3": u_bc3,
            "xyt_bc4": xyt_bc4,
            "u_bc4": u_bc4,
            "xyt_ic": xyt_ic,
            "u_ic": u_ic,
            "mode": (n, m),
        }


def compute_losses(model, batch, nu, weight_pde=0.1, weight_bc=0.1, weight_ic=0.1, device="cuda"):
    """
    Compute all loss components for 2D heat equation.
    """
    ctx_feats = batch["u_sparse"]
    ctx_pos = batch["xyt_sparse"]
    
    # Data loss: match sparse observations
    xyt_sparse_query = batch["xyt_sparse"].clone().requires_grad_(True)
    u_hat_obs = model(ctx_feats, ctx_pos, xyt_sparse_query)
    loss_data = F.mse_loss(u_hat_obs, batch["u_sparse"])
    
    # PDE loss: residual at collocation points
    colloc_query = batch["colloc"].clone().requires_grad_(True)
    u_colloc = model(ctx_feats, ctx_pos, colloc_query)
    residual = model.compute_pde_residual_2d(u_colloc, colloc_query, nu)
    loss_pde = torch.log(1 + (residual ** 2).mean())
    
    # Boundary condition losses (4 edges)
    u_bc1 = model(ctx_feats, ctx_pos, batch["xyt_bc1"])
    u_bc2 = model(ctx_feats, ctx_pos, batch["xyt_bc2"])
    u_bc3 = model(ctx_feats, ctx_pos, batch["xyt_bc3"])
    u_bc4 = model(ctx_feats, ctx_pos, batch["xyt_bc4"])
    
    loss_bc = (
        F.mse_loss(u_bc1, batch["u_bc1"]) +
        F.mse_loss(u_bc2, batch["u_bc2"]) +
        F.mse_loss(u_bc3, batch["u_bc3"]) +
        F.mse_loss(u_bc4, batch["u_bc4"])
    )
    
    # Initial condition loss
    u_ic = model(ctx_feats, ctx_pos, batch["xyt_ic"])
    loss_ic = F.mse_loss(u_ic, batch["u_ic"])
    
    # Total loss
    loss = loss_data + weight_pde * loss_pde + weight_bc * loss_bc + weight_ic * loss_ic
    
    return {
        "loss": loss,
        "data": loss_data.item(),
        "pde": loss_pde.item(),
        "bc": loss_bc.item(),
        "ic": loss_ic.item(),
    }


def training_step(model, optimizer, batch, nu, weight_pde=0.1, weight_bc=0.1, weight_ic=0.1, device="cuda"):
    """Single training step with backpropagation."""
    model.train()
    optimizer.zero_grad()
    
    loss_dict = compute_losses(model, batch, nu, weight_pde, weight_bc, weight_ic, device)
    loss = loss_dict["loss"]
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss_dict


def evaluate_model_2d(model, nu, mode, device, nx=64, ny=64, nt=32, M_eval=100):
    """
    Evaluate model on 2D heat equation.
    
    Args:
        model: PAT model
        nu: diffusivity
        mode: tuple (n, m) for Fourier mode
        device: torch device
        nx, ny, nt: evaluation grid resolution
        M_eval: number of sparse observations for context
    """
    model.eval()
    n, m = mode
    
    with torch.no_grad():
        # Generate sparse observations for THIS specific mode
        x_sparse = torch.rand(1, M_eval, 1, device=device)
        y_sparse = torch.rand(1, M_eval, 1, device=device)
        t_sparse = torch.rand(1, M_eval, 1, device=device)
        u_sparse = exact_solution_2d(
            x_sparse[0, :, :], y_sparse[0, :, :], t_sparse[0, :, :], nu, n=n, m=m
        ).unsqueeze(0)
        xyt_sparse = torch.cat([x_sparse, y_sparse, t_sparse], dim=-1)
        
        # Dense evaluation grid (fixed time slice for visualization)
        t_eval = 0.5
        xe = torch.linspace(0, 1, steps=nx, device=device)
        ye = torch.linspace(0, 1, steps=ny, device=device)
        XE, YE = torch.meshgrid(xe, ye, indexing="ij")
        TE = torch.full_like(XE, t_eval)
        
        XYT_eval = torch.stack([XE, YE, TE], dim=-1).reshape(1, nx * ny, 3)
        
        # Predict using context from THIS mode
        pred = model(u_sparse, xyt_sparse, XYT_eval)
        pred = pred.reshape(nx, ny)
        
        gt = exact_solution_2d(XE, YE, TE, nu, n=n, m=m)
        
        mse = F.mse_loss(pred, gt).item()
        mae = (pred - gt).abs().mean().item()
        mask = gt.abs() > 1e-3
        if mask.sum() > 0:
            rel_error = ((pred - gt).abs() / (gt.abs() + 1e-8))[mask].mean().item()
        else:
            rel_error = float('nan')
        max_error = (pred - gt).abs().max().item()
    
    return {
        "mse": mse,
        "mae": mae,
        "rel_error": rel_error,
        "max_error": max_error,
        "pred": pred.cpu().numpy(),
        "gt": gt.cpu().numpy(),
        "t_eval": t_eval,
    }


def plot_training_progress(history, save_path):
    """Plot training curves."""
    def to_numpy(data):
        if isinstance(data, list):
            return [x.item() if torch.is_tensor(x) else x for x in data]
        return data
    
    steps = to_numpy(history["step"])
    losses = to_numpy(history["loss"])
    data_losses = to_numpy(history["data"])
    pde_losses = to_numpy(history["pde"])
    bc_losses = to_numpy(history["bc"])
    ic_losses = to_numpy(history["ic"])
    lrs = to_numpy(history["lr"])
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Total loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(steps, losses, "b-", linewidth=2, label="Total Loss")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Total Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Loss components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(steps, data_losses, label="Data Loss", linewidth=2)
    ax2.semilogy(steps, pde_losses, label="PDE Loss", linewidth=2)
    ax2.semilogy(steps, bc_losses, label="BC Loss", linewidth=2)
    ax2.semilogy(steps, ic_losses, label="IC Loss", linewidth=2)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Component Loss (log scale)")
    ax2.set_title("Loss Components")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Learning rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, lrs, "r-", linewidth=2)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.grid(True, alpha=0.3)
    
    # Evaluation metrics
    if "eval_step" in history and len(history["eval_step"]) > 0:
        eval_steps = to_numpy(history["eval_step"])
        eval_mse = to_numpy(history["eval_mse"])
        eval_rel = to_numpy(history["eval_rel"])
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.semilogy(eval_steps, eval_mse, "go-", linewidth=2, markersize=6, label="Train MSE")
        if "test_mse" in history and len(history["test_mse"]) > 0:
            test_steps = to_numpy(history["test_step"])
            test_mse = to_numpy(history["test_mse"])
            ax4.semilogy(test_steps, test_mse, "ro-", linewidth=2, markersize=6, label="Test MSE")
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("MSE (log scale)")
        ax4.set_title("Evaluation MSE")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(eval_steps, [r * 100 for r in eval_rel], "go-", linewidth=2, markersize=6, label="Train Rel Error")
        if "test_rel" in history and len(history["test_rel"]) > 0:
            test_rel = to_numpy(history["test_rel"])
            ax5.plot(test_steps, [r * 100 for r in test_rel], "ro-", linewidth=2, markersize=6, label="Test Rel Error")
        ax5.set_xlabel("Training Step")
        ax5.set_ylabel("Relative Error (%)")
        ax5.set_title("Evaluation Relative Error")
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def plot_solution_comparison_2d(eval_results, save_path, mode=None):
    """Plot 2D solution comparison."""
    pred = eval_results["pred"]
    gt = eval_results["gt"]
    error = np.abs(pred - gt)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())
    
    im1 = axes[0].imshow(pred.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Prediction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(gt.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(error.T, origin="lower", aspect="auto", cmap="hot")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im3, ax=axes[2])
    
    if mode:
        fig.suptitle(f"2D Heat Equation - Mode ({mode[0]},{mode[1]}) at t={eval_results['t_eval']:.2f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Solution comparison plot saved to {save_path}")


def train_pat_2d(args):
    """Main training loop for 2D heat equation."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse modes from string format "n,m"
    def parse_mode(mode_str):
        return tuple(map(int, mode_str.split(",")))
    
    train_modes = [parse_mode(m) for m in args.train_modes]
    test_modes = [parse_mode(m) for m in args.test_modes]
    
    print(f"Training modes: {train_modes}")
    print(f"Test modes: {test_modes}")
    
    # Create dataset
    dataset = SparseHeat2DDataset(
        num_instances=10**6,
        M_sparse=args.M,
        nu=args.nu,
        modes=train_modes,
        Nc=args.Nc,
        Nbc=args.Nbc,
        Nic=args.Nic,
        device=device,
    )
    
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    
    # Configure model for 2D problem
    cfg = PATConfig()
    cfg.d_patch = 1  # Scalar field
    cfg.d_pos = 3  # (x, y, t) positions
    cfg.n_layer = args.n_layers
    cfg.n_head = args.n_heads
    cfg.n_embd = args.n_embd
    cfg.dropout = args.dropout
    cfg.use_gradient_checkpointing = args.use_checkpointing
    cfg.alpha = args.alpha
    cfg.nu_bar = args.nu
    
    model = PATModel(cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        "step": [],
        "loss": [],
        "data": [],
        "pde": [],
        "bc": [],
        "ic": [],
        "lr": [],
        "eval_step": [],
        "eval_mse": [],
        "eval_mae": [],
        "eval_rel": [],
        "test_step": [],
        "test_mse": [],
        "test_mae": [],
        "test_rel": [],
    }
    
    print("Starting training...")
    print("-" * 80)
    
    best_eval_mse = float("inf")
    data_iter = iter(loader)
    
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        stats = training_step(
            model, optimizer, batch, args.nu, args.weight_pde, args.weight_bc, args.weight_ic, device
        )
        scheduler.step()
        
        if step % args.print_every == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            
            print(
                f"[{step:05d}] loss={stats['loss']:.3e} | "
                f"data={stats['data']:.3e} pde={stats['pde']:.3e} "
                f"bc={stats['bc']:.3e} ic={stats['ic']:.3e} | "
                f"lr={current_lr:.2e}"
            )
            
            history["step"].append(step)
            history["loss"].append(stats["loss"])
            history["data"].append(stats["data"])
            history["pde"].append(stats["pde"])
            history["bc"].append(stats["bc"])
            history["ic"].append(stats["ic"])
            history["lr"].append(current_lr)
        
        if step % args.eval_every == 0 or step == args.steps:
            eval_results = evaluate_model_2d(
                model, args.nu, train_modes[0], device, M_eval=args.M
            )
            
            print(
                f"       EVAL (seen mode {train_modes[0]}) → MSE={eval_results['mse']:.3e} "
                f"MAE={eval_results['mae']:.3e} "
                f"RelErr={eval_results['rel_error']:.3%}"
            )
            
            history["eval_step"].append(step)
            history["eval_mse"].append(eval_results["mse"])
            history["eval_mae"].append(eval_results["mae"])
            history["eval_rel"].append(eval_results["rel_error"])
            
            # Test set evaluation (unseen mode)
            test_results = evaluate_model_2d(
                model, args.nu, test_modes[0], device, M_eval=args.M
            )
            
            print(
                f"       TEST (unseen mode {test_modes[0]}) → MSE={test_results['mse']:.3e} "
                f"MAE={test_results['mae']:.3e} "
                f"RelErr={test_results['rel_error']:.3%}"
            )
            
            history["test_step"].append(step)
            history["test_mse"].append(test_results["mse"])
            history["test_mae"].append(test_results["mae"])
            history["test_rel"].append(test_results["rel_error"])
            
            if eval_results["mse"] < best_eval_mse:
                best_eval_mse = eval_results["mse"]
                print(f"       → New best MSE: {best_eval_mse:.3e}")
                
                if args.save_path:
                    best_path = args.save_path.replace(".pt", "_best.pt")
                    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
                    checkpoint = {
                        "model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "step": step,
                        "mse": best_eval_mse,
                        "args": vars(args),
                    }
                    torch.save(checkpoint, best_path)
            
            print("-" * 80)
        
        if step % args.save_every == 0 and args.save_path:
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg.__dict__,
                "step": step,
                "history": history,
                "args": vars(args),
            }
            torch.save(checkpoint, args.save_path)
            print(f"Checkpoint saved to {args.save_path}")
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    print("\n--- Interpolation (Seen Modes) ---")
    for mode in train_modes:
        eval_seen = evaluate_model_2d(
            model, args.nu, mode, device, nx=128, ny=128, nt=64, M_eval=args.M
        )
        print(f"Mode {mode}: MSE={eval_seen['mse']:.4e}, MAE={eval_seen['mae']:.4e}, "
              f"RelErr={eval_seen['rel_error']:.4e}")
    
    print("\n--- Extrapolation (Unseen Modes) ---")
    for mode in test_modes:
        eval_unseen = evaluate_model_2d(
            model, args.nu, mode, device, nx=128, ny=128, nt=64, M_eval=args.M
        )
        print(f"Mode {mode}: MSE={eval_unseen['mse']:.4e}, MAE={eval_unseen['mae']:.4e}, "
              f"RelErr={eval_unseen['rel_error']:.4e}")
    
    final_eval = evaluate_model_2d(
        model, args.nu, train_modes[0], device, nx=128, ny=128, nt=64, M_eval=args.M
    )
    
    print(f"\nBest training MSE: {best_eval_mse:.4e}")
    print("=" * 80)
    
    if args.plot:
        print("\nGenerating visualizations...")
        plot_training_progress(history, "pat_2d_training_progress.png")
        plot_solution_comparison_2d(final_eval, "pat_2d_seen_mode.png", mode=train_modes[0])
        
        unseen_eval = evaluate_model_2d(
            model, args.nu, test_modes[0], device, nx=128, ny=128, nt=64, M_eval=args.M
        )
        plot_solution_comparison_2d(unseen_eval, "pat_2d_unseen_mode.png", mode=test_modes[0])
    
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        final_checkpoint = {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "step": args.steps,
            "mse": final_eval["mse"],
            "args": vars(args),
            "history": history,
        }
        final_path = args.save_path.replace(".pt", "_final.pt")
        torch.save(final_checkpoint, final_path)
        print(f"\nFinal model saved to {final_path}")
    
    print("\nTraining complete!")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Train PAT for 2D sparse reconstruction")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)
    
    ap.add_argument("--M", type=int, default=100, help="Number of sparse observations")
    ap.add_argument("--nu", type=float, default=0.1, help="Diffusivity")
    ap.add_argument("--train_modes", type=str, nargs="+", default=["1,1", "1,2"], 
                    help="Training modes as 'n,m' strings")
    ap.add_argument("--test_modes", type=str, nargs="+", default=["2,2", "2,3"], 
                    help="Test modes as 'n,m' strings")
    ap.add_argument("--alpha", type=float, default=1.0, help="Physics guidance strength (0=off)")
    
    ap.add_argument("--Nc", type=int, default=4096, help="Number of collocation points")
    ap.add_argument("--Nbc", type=int, default=512, help="Number of BC points per edge")
    ap.add_argument("--Nic", type=int, default=1024, help="Number of IC points")
    
    ap.add_argument("--steps", type=int, default=10000, help="Training steps")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    
    ap.add_argument("--weight_pde", type=float, default=0.1, help="PDE loss weight")
    ap.add_argument("--weight_bc", type=float, default=0.1, help="BC loss weight")
    ap.add_argument("--weight_ic", type=float, default=0.1, help="IC loss weight")
    
    ap.add_argument("--n_layers", type=int, default=6, help="Transformer layers")
    ap.add_argument("--n_heads", type=int, default=8, help="Attention heads")
    ap.add_argument("--n_embd", type=int, default=256, help="Embedding dimension")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_checkpointing", action="store_true", help="Gradient checkpointing")
    
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--save_path", type=str, default="checkpoints/pat_2d_sparse.pt")
    ap.add_argument("--plot", action="store_true", help="Generate plots")
    
    args = ap.parse_args()
    
    set_seed(args.seed)
    train_pat_2d(args)


if __name__ == "__main__":
    main()
