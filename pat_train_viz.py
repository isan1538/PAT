# pat_train.py - CUDA-OPTIMIZED VERSION
"""
CUDA optimizations:
- Reduced default batch size for memory efficiency
- Pre-allocated static training data (reused across iterations)
- Deferred logging to avoid GPU-CPU sync overhead
- Gradient accumulation support for effective large batch sizes
- Better memory management with torch.cuda.empty_cache()
- Mixed precision training support (optional)
"""

import math, random, argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pat_model_1207 import PATConfig, PATModel

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ----- exact solution for synthetic 1D diffusion test -----
def exact_solution(x, t, nu, n=1):
    """u(x,t) = exp(-nu*(n*pi)^2 * t) * sin(n*pi*x), x,t in [0,1]"""
    return torch.exp(-nu*(n*math.pi)**2 * t) * torch.sin(n*math.pi * x)

# ----- build LR grid and boxes -----
def make_lr_grid(nx=16, nt=16):
    """Create spatial-temporal grid of patches."""
    xs = torch.linspace(0, 1, steps=nx+1)
    ts = torch.linspace(0, 1, steps=nt+1)
    x_centers = 0.5 * (xs[:-1] + xs[1:])
    t_centers = 0.5 * (ts[:-1] + ts[1:])
    dx = (xs[1]-xs[0]).item()
    dt = (ts[1]-ts[0]).item()
    xc, tc = torch.meshgrid(x_centers, t_centers, indexing="ij")
    centers = torch.stack([xc, tc], dim=-1).reshape(-1, 2)  # (P,2)
    X0, X1 = (xc-dx/2).reshape(-1), (xc+dx/2).reshape(-1)
    T0, T1 = (tc-dt/2).reshape(-1), (tc+dt/2).reshape(-1)
    boxes = torch.stack([X0, X1, T0, T1], dim=-1)          # (P,4)
    return centers, boxes, dx, dt

# ----- Generate quadrature points for box averaging -----
def generate_quadrature_points(boxes, nq=3, device='cpu'):
    """
    Generate Gauss-Legendre quadrature points for each box.
    Returns: quad_points (P, nq^2, 2), weights (P, nq^2)
    """
    if nq == 3:
        nodes = torch.tensor([-math.sqrt(3/5), 0.0, math.sqrt(3/5)], dtype=torch.float32)
        weights = torch.tensor([5/9, 8/9, 5/9], dtype=torch.float32)
    elif nq == 2:
        nodes = torch.tensor([-1/math.sqrt(3), 1/math.sqrt(3)], dtype=torch.float32)
        weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    else:
        raise ValueError("nq must be 2 or 3")
    nodes = nodes.to(device)
    weights = weights.to(device)
    P = boxes.size(0)
    x0, x1, t0, t1 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    
    # Map from [-1,1] to [x0,x1] and [t0,t1]
    xq = 0.5 * ((x1 - x0).unsqueeze(1) * nodes.unsqueeze(0) + (x1 + x0).unsqueeze(1))  # (P,nq)
    tq = 0.5 * ((t1 - t0).unsqueeze(1) * nodes.unsqueeze(0) + (t1 + t0).unsqueeze(1))  # (P,nq)
    
    # Create 2D grid of quadrature points
    xqq = xq.unsqueeze(2).expand(P, nq, nq)  # (P,nq,nq)
    tqq = tq.unsqueeze(1).expand(P, nq, nq)  # (P,nq,nq)
    
    X = xqq.reshape(P, nq*nq)  # (P,nq^2)
    T = tqq.reshape(P, nq*nq)  # (P,nq^2)
    
    quad_points = torch.stack([X, T], dim=-1).to(device)  # (P,nq^2,2)
    
    # Compute weights (tensor product)
    W2 = (weights.unsqueeze(1) @ weights.unsqueeze(0)).reshape(-1)  # (nq^2,)
    W = W2.unsqueeze(0).expand(P, -1).to(device)  # (P,nq^2)
    
    # Jacobian for coordinate transformation
    J = 0.25 * (x1 - x0) * (t1 - t0)  # (P,)
    
    return quad_points, W, J.to(device)

# ----- Compute true LR averages using quadrature -----
def compute_lr_averages(boxes, quad_points, weights, jacobians, exact_fn, device='cpu'):
    """
    Compute box averages of exact solution using quadrature.
    Returns: y_lr (P, 1)
    """
    P = boxes.size(0)
    nq2 = quad_points.size(1)
    
    # Evaluate exact solution at all quadrature points
    x_flat = quad_points[:, :, 0].reshape(P * nq2, 1)
    t_flat = quad_points[:, :, 1].reshape(P * nq2, 1)
    u_flat = exact_fn(x_flat, t_flat)  # (P*nq^2, 1)
    u_quad = u_flat.reshape(P, nq2, 1)  # (P, nq^2, 1)
    
    # Compute weighted average
    x0, x1, t0, t1 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x1 - x0) * (t1 - t0)  # (P,)
    
    # Quadrature sum: sum_i w_i * J * u_i
    weighted_sum = (u_quad * (weights.unsqueeze(-1) * jacobians.unsqueeze(1).unsqueeze(-1))).sum(dim=1)  # (P,1)
    avg = weighted_sum / area.unsqueeze(-1)  # (P,1)
    
    return avg.to(device)

# ----- Simple patch features based on actual LR data -----
def build_patch_features(u_lr_vals, nx, nt):
    """
    Build simple statistical features from LR measurements.
    u_lr_vals: (B, P, 1) - actual box-averaged values
    """
    B, P, _ = u_lr_vals.shape
    u_grid = u_lr_vals.reshape(B, nx, nt, 1)
    
    # Feature 0: Raw value
    feat0 = u_grid
    
    # Feature 1: Temporal mean (broadcast)
    mu_t = u_grid.mean(dim=2, keepdim=True)
    feat1 = mu_t.expand(-1, -1, nt, -1)
    
    # Feature 2: Spatial mean (broadcast)
    mu_x = u_grid.mean(dim=1, keepdim=True)
    feat2 = mu_x.expand(-1, nx, -1, -1)
    
    # Feature 3: Global mean
    mu_global = u_grid.mean(dim=(1,2), keepdim=True)
    feat3 = mu_global.expand(-1, nx, nt, -1)
    
    feats = torch.cat([feat0, feat1, feat2, feat3], dim=-1).reshape(B, P, -1)  # (B,P,4)
    return feats

def plot_training_metrics(history, save_path='training_metrics.png'):
    """
    Plot training metrics: loss components, learning rate, and error trends.
    history: dict with keys 'step', 'loss', 'data', 'pde', 'bc', 'ic', 'lr'
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    steps = history['step']
    
    # Plot 1: Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(steps, history['loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Total Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Loss Components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(steps, history['data'], label='Data Loss', linewidth=2)
    ax2.semilogy(steps, history['pde'], label='PDE Loss', linewidth=2)
    ax2.semilogy(steps, history['bc'], label='BC Loss', linewidth=2)
    ax2.semilogy(steps, history['ic'], label='IC Loss', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Component Loss (log scale)')
    ax2.set_title('Loss Components')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Learning Rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, history['lr'], 'r-', linewidth=2)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss Ratio Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    data_arr = np.array(history['data'])
    pde_arr = np.array(history['pde'])
    bc_arr = np.array(history['bc'])
    ic_arr = np.array(history['ic'])
    total = data_arr + pde_arr + bc_arr + ic_arr + 1e-10
    
    ax4.plot(steps, data_arr / total, label='Data %', linewidth=2)
    ax4.plot(steps, pde_arr / total, label='PDE %', linewidth=2)
    ax4.plot(steps, bc_arr / total, label='BC %', linewidth=2)
    ax4.plot(steps, ic_arr / total, label='IC %', linewidth=2)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Loss Contribution (%)')
    ax4.set_title('Relative Loss Contributions')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Data vs Physics Loss
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogy(steps, data_arr, label='Data Loss', linewidth=2)
    physics_loss = pde_arr + bc_arr + ic_arr
    ax5.semilogy(steps, physics_loss, label='Physics Loss (PDE+BC+IC)', linewidth=2)
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('Loss (log scale)')
    ax5.set_title('Data vs Physics Loss')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Loss Stability (moving std)
    ax6 = fig.add_subplot(gs[2, 1])
    window = min(50, len(steps) // 10)
    if len(steps) > window:
        loss_arr = np.array(history['loss'])
        moving_std = np.array([np.std(loss_arr[max(0, i-window):i+1]) 
                               for i in range(len(loss_arr))])
        ax6.semilogy(steps, moving_std, 'g-', linewidth=2)
        ax6.set_xlabel('Training Step')
        ax6.set_ylabel('Loss Std Dev (log scale)')
        ax6.set_title(f'Training Stability (window={window})')
        ax6.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training metrics saved to {save_path}")

def plot_final_predictions(x_hr, y_hr, y_pred, exact_fn, nu, n, save_path='final_predictions.png'):
    """
    Plot final predictions vs ground truth at HR points.
    """
    # Move to CPU and convert to numpy (detach if requires_grad)
    x_hr_np = x_hr.detach().cpu().numpy()
    y_hr_np = y_hr.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Scatter plot in space-time (colored by value)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(x_hr_np[:, 0], x_hr_np[:, 1], c=y_hr_np, 
                          cmap='viridis', s=10, alpha=0.6)
    ax1.set_xlabel('x (space)')
    ax1.set_ylabel('t (time)')
    ax1.set_title('Ground Truth HR Points')
    plt.colorbar(scatter1, ax=ax1, label='u(x,t)')
    
    # Plot 2: Predicted values
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(x_hr_np[:, 0], x_hr_np[:, 1], c=y_pred_np, 
                          cmap='viridis', s=10, alpha=0.6)
    ax2.set_xlabel('x (space)')
    ax2.set_ylabel('t (time)')
    ax2.set_title('Predicted HR Points')
    plt.colorbar(scatter2, ax=ax2, label='u(x,t)')
    
    # Plot 3: Prediction vs Ground Truth
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_hr_np, y_pred_np, alpha=0.5, s=10)
    min_val = min(y_hr_np.min(), y_pred_np.min())
    max_val = max(y_hr_np.max(), y_pred_np.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Ground Truth')
    ax3.set_ylabel('Prediction')
    ax3.set_title('Prediction vs Ground Truth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax4 = fig.add_subplot(gs[1, 1])
    errors = (y_pred_np - y_hr_np).flatten()
    ax4.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution (mean={errors.mean():.4f}, std={errors.std():.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Final predictions saved to {save_path}")

def plot_solution_comparison(model, ctx_feats, ctx_pos, nu, n, device, save_path='solution_comparison.png'):
    """
    Plot full solution field: ground truth vs prediction vs error.
    """
    model.eval()
    with torch.no_grad():
        # Create dense evaluation grid
        nx_eval, nt_eval = 128, 64
        xe = torch.linspace(0, 1, steps=nx_eval, device=device)
        te = torch.linspace(0, 1, steps=nt_eval, device=device)
        XE, TE = torch.meshgrid(xe, te, indexing="ij")
        
        # Evaluate model
        XT_eval = torch.stack([XE, TE], dim=-1).reshape(1, nx_eval*nt_eval, 2)
        pred = model(ctx_feats[:1], ctx_pos[:1], XT_eval)["u"]
        pred = pred.reshape(nx_eval, nt_eval).cpu().numpy()
        
        # Ground truth
        gt = exact_solution(XE, TE, nu, n=n).cpu().numpy()
        
        # Error
        error = pred - gt
        
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Ground Truth
    im1 = axes[0, 0].imshow(gt, aspect='auto', origin='lower', 
                            extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 0].set_xlabel('t (time)')
    axes[0, 0].set_ylabel('x (space)')
    axes[0, 0].set_title('Ground Truth u(x,t)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Prediction
    im2 = axes[0, 1].imshow(pred, aspect='auto', origin='lower', 
                            extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 1].set_xlabel('t (time)')
    axes[0, 1].set_ylabel('x (space)')
    axes[0, 1].set_title('Prediction û(x,t)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Error
    im3 = axes[1, 0].imshow(error, aspect='auto', origin='lower', 
                            extent=[0, 1, 0, 1], cmap='RdBu_r')
    axes[1, 0].set_xlabel('t (time)')
    axes[1, 0].set_ylabel('x (space)')
    axes[1, 0].set_title('Error (û - u)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Temporal slices at different spatial locations
    x_slices = [0.25, 0.5, 0.75]
    x_indices = [int(x * nx_eval) for x in x_slices]
    for idx, x_val in zip(x_indices, x_slices):
        axes[1, 1].plot(te.cpu().numpy(), gt[idx, :], '--', 
                       label=f'GT x={x_val:.2f}', alpha=0.7)
        axes[1, 1].plot(te.cpu().numpy(), pred[idx, :], '-', 
                       label=f'Pred x={x_val:.2f}')
    axes[1, 1].set_xlabel('t (time)')
    axes[1, 1].set_ylabel('u(x,t)')
    axes[1, 1].set_title('Temporal Slices at Different x')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Solution comparison saved to {save_path}")

def main():
    ap = argparse.ArgumentParser(description="Train PAT for 1D diffusion (CUDA-Optimized)")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)
    
    # CUDA FIX: Reduced default batch size for memory efficiency
    ap.add_argument("--batch_size", type=int, default=4, help="Training batch size (reduced for CUDA)")
    ap.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    
    ap.add_argument("--nu", type=float, default=0.1, help="diffusivity")
    ap.add_argument("--mode_n", type=int, default=1, help="sine mode n in sin(n*pi*x)")
    ap.add_argument("--nx_lr", type=int, default=16, help="LR grid x resolution")
    ap.add_argument("--nt_lr", type=int, default=16, help="LR grid t resolution")
    
    # CUDA FIX: Reduced default sizes for better memory usage
    ap.add_argument("--n_hr", type=int, default=1024, help="Number of HR data points (reduced)")
    ap.add_argument("--n_colloc", type=int, default=2048, help="Number of collocation points (reduced)")
    
    ap.add_argument("--steps", type=int, default=1500, help="Training steps")
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500, help="Checkpoint save frequency")
    
    # CUDA FIX: Increased gradient clipping for stability
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping norm (increased)")
    ap.add_argument("--lr_decay", type=float, default=0.99, help="LR decay per 100 steps")
    
    ap.add_argument("--save_ckpt", type=str, default="pat_ckpt.pt")
    ap.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")
    
    # CUDA FIX: Mixed precision training option
    ap.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    ap.add_argument("--empty_cache_freq", type=int, default=100, help="Clear CUDA cache every N steps")
    ap.add_argument("--no_uncertainty", action="store_true", help="Disable uncertainty weighting (more stable)")
    ap.add_argument("--PAT", action="store_true", help="Use PAT model")
    
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1) LR grid and quadrature points
    centers, boxes, dx, dt = make_lr_grid(args.nx_lr, args.nt_lr)
    P = centers.size(0)
    print(f"LR grid: {args.nx_lr}x{args.nt_lr} = {P} patches")
    centers = centers.to(device)
    boxes   = boxes.to(device)
    
    # Generate quadrature points for the model
    quad_points, weights, jacobians = generate_quadrature_points(boxes, nq=3, device=device)
    print(f"Quadrature: {quad_points.shape[1]} points per patch")

    # 2) Generate synthetic LR data using proper box averaging
    B = args.batch_size
    
    # Compute true box-averaged LR data
    exact_fn = lambda x, t: exact_solution(x, t, args.nu, n=args.mode_n)
    y_lr_single = compute_lr_averages(boxes, quad_points, weights, jacobians, exact_fn, device=device)
    y_lr = y_lr_single.unsqueeze(0).expand(B, -1, -1)  # (B, P, 1)
    
    print(f"LR data range: [{y_lr.min().item():.4f}, {y_lr.max().item():.4f}]")

    # CUDA FIX: Pre-allocate all training data (static, reused across iterations)
    print("Pre-allocating training data...")
    
    # 3) Generate HR training data
    x_hr = torch.rand(B, args.n_hr, 1, device=device)
    t_hr = torch.rand(B, args.n_hr, 1, device=device)
    y_hr = exact_solution(x_hr, t_hr, args.nu, n=args.mode_n)
    x_hr = torch.cat([x_hr, t_hr], dim=-1)  # (B, n_hr, 2)
    
    print(f"HR data: {args.n_hr} points, range: [{y_hr.min().item():.4f}, {y_hr.max().item():.4f}]")

    # 4) Generate PDE collocation points
    x_r = torch.rand(B, args.n_colloc, 1, device=device)
    t_r = torch.rand(B, args.n_colloc, 1, device=device)
    colloc = torch.cat([x_r, t_r], dim=-1)  # (B, n_colloc, 2)

    # 5) Boundary conditions
    Na = Nb = 256
    ta = torch.rand(B, Na, 1, device=device)
    tb = torch.rand(B, Nb, 1, device=device)
    ga = exact_solution(torch.zeros_like(ta), ta, args.nu, n=args.mode_n)
    gb = exact_solution(torch.ones_like(tb), tb, args.nu, n=args.mode_n)

    # 6) Initial condition
    N0 = 512
    x0 = torch.rand(B, N0, 1, device=device)
    u0 = exact_solution(x0, torch.zeros_like(x0), args.nu, n=args.mode_n)

    # 7) Context features from actual LR data
    ctx_feats = build_patch_features(y_lr, args.nx_lr, args.nt_lr)
    ctx_pos = centers.view(1, P, 2).expand(B, -1, -1).to(device)
    
    print(f"Context features: {ctx_feats.shape}, pos: {ctx_pos.shape}")

    # 8) Prepare quadrature points for model (add batch dimension)
    quad_points_batch = quad_points.unsqueeze(0).expand(B, -1, -1, -1)  # (B, P, nq^2, 2)

    # 9) Initialize model
    cfg = PATConfig()
    cfg.d_patch = ctx_feats.size(-1)
    cfg.x_min = 0.0
    cfg.x_max = 1.0
    cfg.use_gradient_checkpointing = True  # CUDA FIX: Enable checkpointing

    model = PATModel(cfg).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Gradient checkpointing: {cfg.use_gradient_checkpointing}")

    # 10) Optimizer with learning rate scheduling
    opt = model.configure_optimizers(
        weight_decay=args.weight_decay, 
        learning_rate=args.lr, 
        betas=(0.9, 0.95),
        device_type=("cuda" if device.type == "cuda" else "cpu")
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=args.lr_decay)
    
    # CUDA FIX: Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("Mixed precision training enabled")

    # 11) Resume if requested
    start_step = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 1) + 1
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if args.use_amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"Resumed from {args.resume} at step {start_step}")

    # CUDA FIX: Pre-build training dict (reused, only requires_grad changes)
    if args.PAT: 
        # cfg.use_uncertainty = False
        training_dict_template = {
            "nu": args.nu,
            "hr": {"x": x_hr, "y": y_hr},
            "lr": {"y_lr": y_lr, "quad_points": quad_points_batch},
            "pde": {"colloc": colloc, "f_r": None},
            "bc": {"a_times": ta, "g_a": ga, "b_times": tb, "g_b": gb},
            "ic": {"x0": x0, "u0": u0},
        }
    else:
        cfg.alpha = 0.0  # No PDE loss
        training_dict_template = {
            "nu": args.nu,
            "hr": {"x": x_hr, "y": y_hr},
            "lr": {"y_lr": y_lr, "quad_points": quad_points_batch},
        }
        print("Mode: Pure Transformer")
        print(f"  - Physics-guided attention: OFF (alpha={cfg.alpha})")
        print(f"  - Physics losses: OFF (data only)")
        

    # CUDA FIX: Deferred logging to avoid frequent GPU-CPU sync
    log_buffer = []
    best_loss = float("inf")
    
    # History tracking for visualization
    history = {
        'step': [],
        'loss': [],
        'data': [],
        'pde': [],
        'bc': [],
        'ic': [],
        'lr': []
    }
    
    print(f"\nStarting training (effective batch size: {B * args.accumulation_steps})...")
    print("=" * 80)

    for step in range(start_step, args.steps + 1):
        model.train()
        
        # CUDA FIX: Gradient accumulation
        if (step - 1) % args.accumulation_steps == 0:
            opt.zero_grad(set_to_none=True)

        # Forward pass with optional mixed precision
        with autocast() if args.use_amp else torch.enable_grad():
            out = model(ctx_feats, ctx_pos, x_hr, f_q=None, training_dict=training_dict_template)
            loss = out["loss"] / args.accumulation_steps  # Scale loss for accumulation
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nNaN loss detected at step {step}! Stopping.")
            break
        
        # Backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (only after accumulation_steps)
        if step % args.accumulation_steps == 0:
            if args.use_amp:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            scheduler.step()

        # CUDA FIX: Deferred logging (collect metrics, print in batch)
        if step % args.print_every == 0 or step == 1:
            comps = out["loss_components"]
            lr_current = scheduler.get_last_lr()[0]
            actual_loss = loss.item() * args.accumulation_steps  # Unscale for logging
            
            log_entry = {
                'step': step,
                'loss': actual_loss,
                'data': comps['data'].item(),
                'pde': comps['pde'].item(),
                'bc': comps['bc'].item(),
                'ic': comps['ic'].item(),
                'lr': lr_current
            }
            log_buffer.append(log_entry)
            
            # Record to history
            history['step'].append(step)
            history['loss'].append(actual_loss)
            history['data'].append(comps['data'].item())
            history['pde'].append(comps['pde'].item())
            history['bc'].append(comps['bc'].item())
            history['ic'].append(comps['ic'].item())
            history['lr'].append(lr_current)

            
            # Print buffered logs
            for entry in log_buffer:
                print(f"[{entry['step']:04d}] loss={entry['loss']:.4e} | "
                      f"data={entry['data']:.4e} pde={entry['pde']:.4e} "
                      f"bc={entry['bc']:.4e} ic={entry['ic']:.4e} | lr={entry['lr']:.2e}")
            log_buffer.clear()
            
            # Track best model
            if actual_loss < best_loss:
                best_loss = actual_loss
                print(f"       → New best loss: {best_loss:.4e}")
        
        # Extra detailed logging every 50 steps
        if step % 500 == 0:
            u_pred = out['u']
            print(f"       u_pred: min={u_pred.min().item():.4e}, "
                  f"max={u_pred.max().item():.4e}, "
                  f"mean={u_pred.mean().item():.4e}")
            
            if cfg.use_uncertainty:
                print(f"       σ: data={torch.exp(model.logsig_data).item():.4e}, "
                      f"pde={torch.exp(model.logsig_pde).item():.4e}")
            
            # CUDA memory info
            if device.type == "cuda":
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"       CUDA memory: {mem_allocated:.2f}GB allocated, "
                      f"{mem_reserved:.2f}GB reserved")
            print("-" * 80)

        # CUDA FIX: Periodic cache clearing
        if device.type == "cuda" and step % args.empty_cache_freq == 0:
            torch.cuda.empty_cache()

        # Save checkpoint periodically
        if step % args.save_every == 0 or step == args.steps:
            if args.save_ckpt:
                ckpt_data = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "loss": best_loss,
                    "config": vars(args)
                }
                if args.use_amp:
                    ckpt_data["scaler"] = scaler.state_dict()
                torch.save(ckpt_data, args.save_ckpt)
                print(f"Checkpoint saved to {args.save_ckpt}")

    # 13) Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        # Evaluate on fine grid
        nx_eval, nt_eval = 256, 128
        xe = torch.linspace(0, 1, steps=nx_eval, device=device)
        te = torch.linspace(0, 1, steps=nt_eval, device=device)
        XE, TE = torch.meshgrid(xe, te, indexing="ij")
        
        # Use single batch for evaluation
        XT_eval = torch.stack([XE, TE], dim=-1).reshape(1, nx_eval*nt_eval, 2)
        
        # Get predictions
        pred = model(ctx_feats[:1], ctx_pos[:1], XT_eval)["u"]
        pred = pred.reshape(nx_eval, nt_eval)
        
        # Ground truth
        gt = exact_solution(XE, TE, args.nu, n=args.mode_n)
        
        # Compute errors
        mse = F.mse_loss(pred, gt)
        mae = (pred - gt).abs().mean()
        # rel_err = ((pred - gt).abs() / (gt.abs() + 1e-8)).mean()
        
        print(f"Fine grid evaluation ({nx_eval}x{nt_eval}):")
        print(f"  MSE:     {mse.item():.4e}")
        print(f"  MAE:     {mae.item():.4e}")
        # print(f"  Rel Err: {rel_err.item():.4%}")
        print(f"  Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        print(f"  GT range:   [{gt.min().item():.4f}, {gt.max().item():.4f}]")
        
        # Generate predictions at HR points for visualization
        with torch.no_grad():
            hr_pred = model(ctx_feats[:1], ctx_pos[:1], x_hr[:1])["u"]
            hr_pred = hr_pred.squeeze(0)  # Remove batch dimension
            hr_gt = y_hr[0]  # First batch
            hr_coords = x_hr[0]  # First batch

    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # Plot training metrics
    plot_training_metrics(history, save_path='training_metrics.png')
    
    # Plot final predictions at HR points
    plot_final_predictions(hr_coords, hr_gt, hr_pred, exact_solution, args.nu, args.mode_n, 
                          save_path='final_predictions.png')
    # 
    # Plot full solution comparison
    # plot_solution_comparison(model, ctx_feats, ctx_pos, args.nu, args.mode_n, device, 
                            # save_path='solution_comparison.png')

    print("\nTraining complete!")
    
    # CUDA cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("CUDA cache cleared")

if __name__ == "__main__":
    main()