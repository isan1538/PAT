"""
Addressing the loss calculation inconsistency, now error is calculated only in training part
 

Pure transformer model without physics guidance:
python pat_training.py --alpha 0.0 --weight_pde 0.0 --weight_bc 0.0 --weight_ic 0.0


Physics_informed transformer model:
python pat_training.py --M 50 --nu 0.1 --modes 1 2 3 --alpha 1.0 --steps 5000 /
     --weight_pde 0.1 --weight_bc 0.1 --weight_ic 0.1 --save_path checkpoints/pat.pt

     python pat_training.py --M 50 --nu 0.1 --alpha 1.0 --steps 5000 --weight_pde 0.1
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

from pat_model_1208 import PATConfig, PATModel


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def exact_solution(x, t, nu, n=1):
    return torch.exp(-nu * (n * math.pi) ** 2 * t) * torch.sin(n * math.pi * x)


class SparseHeatDataset(Dataset):
    def __init__(
        self,
        num_instances,
        M_sparse,
        nu,
        modes,
        Nc,
        Nbc,
        Nic,
        device="cuda",
    ):
        super().__init__()
        self.num_instances = num_instances
        self.M = M_sparse
        self.nu = nu
        self.modes = modes
        self.Nc = Nc
        self.Nbc = Nbc
        self.Nic = Nic
        self.device = device

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        mode = random.choice(self.modes)

        x_sparse = torch.rand(self.M, 1, device=self.device)
        t_sparse = torch.rand(self.M, 1, device=self.device)
        u_sparse = exact_solution(x_sparse, t_sparse, self.nu, n=mode)
        xt_sparse = torch.cat([x_sparse, t_sparse], dim=-1)

        x_colloc = torch.rand(self.Nc, 1, device=self.device)
        t_colloc = torch.rand(self.Nc, 1, device=self.device)
        colloc = torch.cat([x_colloc, t_colloc], dim=-1)

        ta = torch.rand(self.Nbc, 1, device=self.device)
        ua = exact_solution(torch.zeros_like(ta), ta, self.nu, n=mode)
        xa = torch.cat([torch.zeros_like(ta), ta], dim=-1)

        tb = torch.rand(self.Nbc, 1, device=self.device)
        ub = exact_solution(torch.ones_like(tb), tb, self.nu, n=mode)
        xb = torch.cat([torch.ones_like(tb), tb], dim=-1)

        x0_pts = torch.rand(self.Nic, 1, device=self.device)
        u0 = exact_solution(x0_pts, torch.zeros_like(x0_pts), self.nu, n=mode)
        x0 = torch.cat([x0_pts, torch.zeros_like(x0_pts)], dim=-1)

        return {
            "xt_sparse": xt_sparse,
            "u_sparse": u_sparse,
            "colloc": colloc,
            "xa": xa,
            "ua": ua,
            "xb": xb,
            "ub": ub,
            "x0": x0,
            "u0": u0,
        }


def compute_losses(model, batch, nu, weight_pde=0.1, weight_bc=0.1, weight_ic=0.1, device="cuda"):
    ctx_feats = batch["u_sparse"]
    ctx_pos = batch["xt_sparse"]

    xt_sparse_query = batch["xt_sparse"].clone().requires_grad_(True)
    u_hat_obs = model(ctx_feats, ctx_pos, xt_sparse_query)
    loss_data = F.mse_loss(u_hat_obs, batch["u_sparse"])

    colloc_query = batch["colloc"].clone().requires_grad_(True)
    u_colloc = model(ctx_feats, ctx_pos, colloc_query)
    residual = model.compute_pde_residual(u_colloc, colloc_query, nu)
    # loss_pde = (residual ** 2).mean()
    # loss_pde = F.smooth_l1_loss(residual, torch.zeros_like(residual))
    loss_pde = torch.log(1 + (residual**2).mean())


    u_a = model(ctx_feats, ctx_pos, batch["xa"])
    u_b = model(ctx_feats, ctx_pos, batch["xb"])
    loss_bc_a = F.mse_loss(u_a, batch["ua"])
    loss_bc_b = F.mse_loss(u_b, batch["ub"])
    loss_bc = loss_bc_a + loss_bc_b

    u_ic = model(ctx_feats, ctx_pos, batch["x0"])
    loss_ic = F.mse_loss(u_ic, batch["u0"])

    loss = loss_data + weight_pde * loss_pde + weight_bc * loss_bc + weight_ic * loss_ic

    return {
        "loss": loss,
        "data": loss_data.item(),
        "pde": loss_pde.item(),
        "bc": loss_bc.item(),
        "ic": loss_ic.item(),
    }


def training_step(model, optimizer, batch, nu, weight_pde=0.1, weight_bc=0.1, weight_ic=0.1, device="cuda"):
    model.train()
    optimizer.zero_grad()

    loss_dict = compute_losses(model, batch, nu, weight_pde, weight_bc, weight_ic, device)
    loss = loss_dict["loss"]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_dict


def evaluate_model(model, ctx_feats, ctx_pos, nu, mode, device, nx=128, nt=64):
    model.eval()
    with torch.no_grad():
        xe = torch.linspace(0, 1, steps=nx, device=device)
        te = torch.linspace(0, 1, steps=nt, device=device)
        XE, TE = torch.meshgrid(xe, te, indexing="ij")

        XT_eval = torch.stack([XE, TE], dim=-1).reshape(1, nx * nt, 2)
        pred = model(ctx_feats, ctx_pos, XT_eval)
        pred = pred.reshape(nx, nt)

        gt = exact_solution(XE, TE, nu, n=mode)

        mse = F.mse_loss(pred, gt).item()
        mae = (pred - gt).abs().mean().item()
        mask = gt.abs() > 1e-3  # Only measure where solution is "significant"
        rel_error = ((pred - gt).abs() / (gt.abs() + 1e-8))[mask].mean()        
        max_error = (pred - gt).abs().max().item()

    return {
        "mse": mse,
        "mae": mae,
        "rel_error": rel_error,
        "max_error": max_error,
        "pred": pred.cpu().numpy(),
        "gt": gt.cpu().numpy(),
    }


def plot_training_progress(history, save_path):
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

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(steps, losses, "b-", linewidth=2, label="Total Loss")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Total Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

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

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, lrs, "r-", linewidth=2)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if len(history["eval_step"]) > 0:
        eval_steps = to_numpy(history["eval_step"])
        eval_mses = to_numpy(history["eval_mse"])
        ax4.semilogy(eval_steps, eval_mses, "o-", linewidth=2, markersize=4)
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Evaluation MSE (log scale)")
        ax4.set_title("Validation Performance")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No evaluation data yet", 
                ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Validation Performance")

    ax5 = fig.add_subplot(gs[2, 0])
    data_arr = np.array(data_losses)
    pde_arr = np.array(pde_losses)
    bc_arr = np.array(bc_losses)
    ic_arr = np.array(ic_losses)
    total = data_arr + pde_arr + bc_arr + ic_arr + 1e-10

    ax5.plot(steps, data_arr / total, label="Data %", linewidth=2)
    ax5.plot(steps, pde_arr / total, label="PDE %", linewidth=2)
    ax5.plot(steps, bc_arr / total, label="BC %", linewidth=2)
    ax5.plot(steps, ic_arr / total, label="IC %", linewidth=2)
    ax5.set_xlabel("Training Step")
    ax5.set_ylabel("Loss Contribution (%)")
    ax5.set_title("Relative Loss Contributions")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 1])
    if len(history["eval_step"]) > 1:
        eval_steps = to_numpy(history["eval_step"])
        eval_rels = to_numpy(history["eval_rel"])
        ax6.plot(eval_steps, eval_rels, "g-o", linewidth=2, markersize=4)
        ax6.set_xlabel("Training Step")
        ax6.set_ylabel("Relative Error")
        ax6.set_title("Relative Error Trend")
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "Waiting for more evaluations", 
                ha="center", va="center", transform=ax6.transAxes)
        ax6.set_title("Relative Error Trend")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training progress saved to {save_path}")



def plot_solution_comparison(eval_results, save_path):
    pred = eval_results["pred"]
    gt = eval_results["gt"]
    error = pred - gt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im1 = axes[0, 0].imshow(
        gt, aspect="auto", origin="lower", extent=[0, 1, 0, 1], cmap="viridis"
    )
    axes[0, 0].set_xlabel("t (time)")
    axes[0, 0].set_ylabel("x (space)")
    axes[0, 0].set_title("Ground Truth u(x,t)")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(
        pred, aspect="auto", origin="lower", extent=[0, 1, 0, 1], cmap="viridis"
    )
    axes[0, 1].set_xlabel("t (time)")
    axes[0, 1].set_ylabel("x (space)")
    axes[0, 1].set_title("Prediction û(x,t)")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(
        error, aspect="auto", origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r"
    )
    axes[1, 0].set_xlabel("t (time)")
    axes[1, 0].set_ylabel("x (space)")
    axes[1, 0].set_title("Error (û - u)")
    plt.colorbar(im3, ax=axes[1, 0])

    nx, nt = pred.shape
    x_slices = [0.25, 0.5, 0.75]
    x_indices = [int(x * nx) for x in x_slices]
    t_vals = np.linspace(0, 1, nt)
    for idx, x_val in zip(x_indices, x_slices):
        axes[1, 1].plot(t_vals, gt[idx, :], "--", label=f"GT x={x_val:.2f}", alpha=0.7)
        axes[1, 1].plot(t_vals, pred[idx, :], "-", label=f"Pred x={x_val:.2f}")
    axes[1, 1].set_xlabel("t (time)")
    axes[1, 1].set_ylabel("u(x,t)")
    axes[1, 1].set_title("Temporal Slices at Different x")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Solution comparison saved to {save_path}")


def train_pat_sparse(args):
    device = torch.device(args.device)
    print(f"\n{'=' * 80}")
    print(f"Physics-Aware Transformer - Sparse Reconstruction Training")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    print(f"Sparse points (M): {args.M}")
    print(f"Modes: {args.modes}")
    print(f"Diffusivity (nu): {args.nu}")
    print(f"Physics guidance (alpha): {args.alpha}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"PDE weight: {args.weight_pde}")
    print(f"BC weight: {args.weight_bc}")
    print(f"IC weight: {args.weight_ic}")
    print(f"{'=' * 80}\n")

    dataset = SparseHeatDataset(
        num_instances=10**6,
        M_sparse=args.M,
        nu=args.nu,
        modes=args.modes,
        Nc=args.Nc,
        Nbc=args.Nbc,
        Nic=args.Nic,
        device=device,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )

    cfg = PATConfig()
    cfg.d_patch = 1
    cfg.d_pos = 2
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
    }

    print("Starting training...")
    print("-" * 80)

    best_eval_mse = float("inf")
    data_iter = iter(loader)

    eval_batch = next(data_iter)
    eval_ctx_feats = eval_batch["u_sparse"].to(device)[:1]
    eval_ctx_pos = eval_batch["xt_sparse"].to(device)[:1]

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
            eval_results = evaluate_model(
                model, eval_ctx_feats, eval_ctx_pos, args.nu, args.modes[0], device
            )

            print(
                f"       EVAL → MSE={eval_results['mse']:.3e} "
                f"MAE={eval_results['mae']:.3e} "
                f"RelErr={eval_results['rel_error']:.3%}"
            )

            history["eval_step"].append(step)
            history["eval_mse"].append(eval_results["mse"])
            history["eval_mae"].append(eval_results["mae"])
            history["eval_rel"].append(eval_results["rel_error"])

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
    print("Final Evaluation")
    print("=" * 80)

    final_eval = evaluate_model(
        model, eval_ctx_feats, eval_ctx_pos, args.nu, args.modes[0], device, nx=256, nt=128
    )

    print(f"Final MSE:        {final_eval['mse']:.4e}")
    print(f"Final MAE:        {final_eval['mae']:.4e}")
    print(f"Final Rel Error:  {final_eval['rel_error']:.4%}")
    print(f"Max Error:        {final_eval['max_error']:.4e}")

    if args.plot:
        print("\nGenerating visualizations...")
        plot_training_progress(history, "sparse_training_progress.png")
        plot_solution_comparison(final_eval, "sparse_solution_comparison.png")

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
    ap = argparse.ArgumentParser(description="Train PAT for sparse reconstruction")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--M", type=int, default=50, help="Number of sparse observations")
    ap.add_argument("--nu", type=float, default=0.1, help="Diffusivity")
    ap.add_argument("--modes", type=int, nargs="+", default=[1], help="Sine modes to use")
    ap.add_argument("--alpha", type=float, default=1.0, help="Physics guidance strength (0=off)")

    ap.add_argument("--Nc", type=int, default=2048, help="Number of collocation points")
    ap.add_argument("--Nbc", type=int, default=256, help="Number of BC points")
    ap.add_argument("--Nic", type=int, default=512, help="Number of IC points")

    ap.add_argument("--steps", type=int, default=5000, help="Training steps")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)

    ap.add_argument("--weight_pde", type=float, default=0.1, help="PDE loss weight")
    ap.add_argument("--weight_bc", type=float, default=0.1, help="BC loss weight")
    ap.add_argument("--weight_ic", type=float, default=0.1, help="IC loss weight")

    ap.add_argument("--n_layers", type=int, default=4, help="Transformer layers")
    ap.add_argument("--n_heads", type=int, default=8, help="Attention heads")
    ap.add_argument("--n_embd", type=int, default=256, help="Embedding dimension")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_checkpointing", action="store_true", help="Gradient checkpointing")

    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--save_path", type=str, default="checkpoints/pat_sparse.pt")
    ap.add_argument("--plot", action="store_true", help="Generate plots")

    args = ap.parse_args()

    set_seed(args.seed)
    train_pat_sparse(args)


if __name__ == "__main__":
    main()