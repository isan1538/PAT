#!/usr/bin/env python3

'''
python 1212_deeponet_2d.py \
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
    --hidden_dim 256 \
    --num_layers 6 \
    --print_every 100 \
    --eval_every 500 \
    --plot
    
'''

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


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def exact_solution_2d(x, y, t, nu, n=1, m=1):
    factor = nu * (math.pi ** 2) * (n**2 + m**2)
    return torch.exp(-factor * t) * torch.sin(n * math.pi * x) * torch.sin(m * math.pi * y)


class PIDeepONet2D(nn.Module):
    def __init__(self, branch_dim=128, trunk_dim=128, num_layers=4):
        super().__init__()
        
        self.branch = nn.Sequential(
            nn.Linear(4, branch_dim),
            nn.Tanh(),
            *[layer for _ in range(num_layers-1) 
              for layer in (nn.Linear(branch_dim, branch_dim), nn.Tanh())],
            nn.Linear(branch_dim, branch_dim)
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_dim),
            nn.Tanh(),
            *[layer for _ in range(num_layers-1) 
              for layer in (nn.Linear(trunk_dim, trunk_dim), nn.Tanh())],
            nn.Linear(trunk_dim, trunk_dim)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, xyt_sparse, u_sparse, xyt_query):
        sparse_input = torch.cat([xyt_sparse, u_sparse], dim=-1)
        branch_out = self.branch(sparse_input).mean(dim=1, keepdim=True)
        trunk_out = self.trunk(xyt_query)
        u_pred = (branch_out * trunk_out).sum(dim=-1, keepdim=True) + self.bias
        return u_pred
    
    def compute_pde_residual_2d(self, u, pos, nu):
        u_t = torch.autograd.grad(
            u, pos, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0][..., 2:3]
        
        grads = torch.autograd.grad(
            u, pos, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = grads[..., 0:1]
        u_y = grads[..., 1:2]
        
        u_xx = torch.autograd.grad(
            u_x, pos, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][..., 0:1]
        
        u_yy = torch.autograd.grad(
            u_y, pos, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0][..., 1:2]
        
        residual = u_t - nu * (u_xx + u_yy)
        return residual


class SparseHeat2DDataset(Dataset):
    def __init__(self, num_instances, M_sparse, nu, modes, Nc, Nbc, Nic, device="cuda"):
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
        n, m = random.choice(self.modes)

        x_sparse = torch.rand(self.M, 1, device=self.device)
        y_sparse = torch.rand(self.M, 1, device=self.device)
        t_sparse = torch.rand(self.M, 1, device=self.device)
        u_sparse = exact_solution_2d(x_sparse, y_sparse, t_sparse, self.nu, n=n, m=m)
        xyt_sparse = torch.cat([x_sparse, y_sparse, t_sparse], dim=-1)

        x_colloc = torch.rand(self.Nc, 1, device=self.device)
        y_colloc = torch.rand(self.Nc, 1, device=self.device)
        t_colloc = torch.rand(self.Nc, 1, device=self.device)
        colloc = torch.cat([x_colloc, y_colloc, t_colloc], dim=-1)

        y_bc1 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc1 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc1 = exact_solution_2d(torch.zeros_like(y_bc1), y_bc1, t_bc1, self.nu, n=n, m=m)
        xyt_bc1 = torch.cat([torch.zeros_like(y_bc1), y_bc1, t_bc1], dim=-1)

        y_bc2 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc2 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc2 = exact_solution_2d(torch.ones_like(y_bc2), y_bc2, t_bc2, self.nu, n=n, m=m)
        xyt_bc2 = torch.cat([torch.ones_like(y_bc2), y_bc2, t_bc2], dim=-1)

        x_bc3 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc3 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc3 = exact_solution_2d(x_bc3, torch.zeros_like(x_bc3), t_bc3, self.nu, n=n, m=m)
        xyt_bc3 = torch.cat([x_bc3, torch.zeros_like(x_bc3), t_bc3], dim=-1)

        x_bc4 = torch.rand(self.Nbc, 1, device=self.device)
        t_bc4 = torch.rand(self.Nbc, 1, device=self.device)
        u_bc4 = exact_solution_2d(x_bc4, torch.ones_like(x_bc4), t_bc4, self.nu, n=n, m=m)
        xyt_bc4 = torch.cat([x_bc4, torch.ones_like(x_bc4), t_bc4], dim=-1)

        x_ic = torch.rand(self.Nic, 1, device=self.device)
        y_ic = torch.rand(self.Nic, 1, device=self.device)
        u_ic = exact_solution_2d(x_ic, y_ic, torch.zeros_like(x_ic), self.nu, n=n, m=m)
        xyt_ic = torch.cat([x_ic, y_ic, torch.zeros_like(x_ic)], dim=-1)

        return {
            "xyt_sparse": xyt_sparse, "u_sparse": u_sparse, "colloc": colloc,
            "xyt_bc1": xyt_bc1, "u_bc1": u_bc1, "xyt_bc2": xyt_bc2, "u_bc2": u_bc2,
            "xyt_bc3": xyt_bc3, "u_bc3": u_bc3, "xyt_bc4": xyt_bc4, "u_bc4": u_bc4,
            "xyt_ic": xyt_ic, "u_ic": u_ic
        }


def compute_losses(model, batch, nu, weight_pde=0.1, weight_bc=0.1, weight_ic=0.1, device="cuda"):
    ctx_feats = batch["u_sparse"]
    ctx_pos = batch["xyt_sparse"]

    xyt_sparse_query = batch["xyt_sparse"].clone().requires_grad_(True)
    u_hat_obs = model(ctx_pos, ctx_feats, xyt_sparse_query)
    loss_data = F.mse_loss(u_hat_obs, batch["u_sparse"])

    colloc_query = batch["colloc"].clone().requires_grad_(True)
    u_colloc = model(ctx_pos, ctx_feats, colloc_query)
    residual = model.compute_pde_residual_2d(u_colloc, colloc_query, nu)
    loss_pde = torch.log(1 + (residual**2).mean())

    u_bc1 = model(ctx_pos, ctx_feats, batch["xyt_bc1"])
    u_bc2 = model(ctx_pos, ctx_feats, batch["xyt_bc2"])
    u_bc3 = model(ctx_pos, ctx_feats, batch["xyt_bc3"])
    u_bc4 = model(ctx_pos, ctx_feats, batch["xyt_bc4"])
    
    loss_bc = (F.mse_loss(u_bc1, batch["u_bc1"]) + F.mse_loss(u_bc2, batch["u_bc2"]) +
               F.mse_loss(u_bc3, batch["u_bc3"]) + F.mse_loss(u_bc4, batch["u_bc4"]))

    u_ic = model(ctx_pos, ctx_feats, batch["xyt_ic"])
    loss_ic = F.mse_loss(u_ic, batch["u_ic"])

    loss = loss_data + weight_pde * loss_pde + weight_bc * loss_bc + weight_ic * loss_ic

    return {
        "loss": loss, "data": loss_data.item(), "pde": loss_pde.item(),
        "bc": loss_bc.item(), "ic": loss_ic.item()
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


def evaluate_model_2d(model, nu, mode, device, nx=64, ny=64, M_eval=100):
    model.eval()
    n, m = mode
    
    with torch.no_grad():
        xyt_sparse = torch.rand(1, M_eval, 3, device=device)
        u_sparse = exact_solution_2d(
            xyt_sparse[0, :, 0], xyt_sparse[0, :, 1], xyt_sparse[0, :, 2], nu, n=n, m=m
        ).unsqueeze(0).unsqueeze(-1)
        
        t_eval = 0.5
        xe = torch.linspace(0, 1, steps=nx, device=device)
        ye = torch.linspace(0, 1, steps=ny, device=device)
        XE, YE = torch.meshgrid(xe, ye, indexing="ij")
        TE = torch.full_like(XE, t_eval)

        XYT_eval = torch.stack([XE, YE, TE], dim=-1).reshape(1, nx * ny, 3)
        pred = model(xyt_sparse, u_sparse, XYT_eval)
        pred = pred.reshape(nx, ny)

        gt = exact_solution_2d(XE, YE, TE, nu, n=n, m=m)

        mse = F.mse_loss(pred, gt).item()
        mae = (pred - gt).abs().mean().item()
        mask = gt.abs() > 1e-3
        rel_error = ((pred - gt).abs() / (gt.abs() + 1e-8))[mask].mean().item() if mask.sum() > 0 else float('nan')
        max_error = (pred - gt).abs().max().item()

    return {
        "mse": mse, "mae": mae, "rel_error": rel_error, "max_error": max_error,
        "pred": pred.cpu().numpy(), "gt": gt.cpu().numpy()
    }


def plot_results_2d(history, eval_results, save_path, mode=None):
    def to_numpy(data):
        if isinstance(data, list):
            return [x.item() if torch.is_tensor(x) else x for x in data]
        return data
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    if len(history['step']) > 0:
        ax1.semilogy(to_numpy(history["step"]), to_numpy(history["loss"]), "b-", linewidth=2)
        ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    if len(history['eval_mse']) > 0:
        ax2.semilogy(to_numpy(history['eval_step']), to_numpy(history['eval_mse']), 'b-o', label='Train MSE', markersize=4)
        if len(history['test_mse']) > 0:
            ax2.semilogy(to_numpy(history['test_step']), to_numpy(history['test_mse']), 'r-o', label='Test MSE', markersize=4)
        ax2.set_xlabel("Step"); ax2.set_ylabel("MSE"); ax2.set_title("MSE"); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    if len(history['eval_rel']) > 0:
        ax3.plot(to_numpy(history['eval_step']), [r*100 for r in to_numpy(history['eval_rel'])], 'b-o', label='Train', markersize=4)
        if len(history['test_rel']) > 0:
            ax3.plot(to_numpy(history['test_step']), [r*100 for r in to_numpy(history['test_rel'])], 'r-o', label='Test', markersize=4)
        ax3.set_xlabel("Step"); ax3.set_ylabel("Rel Error (%)"); ax3.set_title("Relative Error"); ax3.legend(); ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(eval_results['gt'].T, origin='lower', aspect='auto', cmap='viridis', extent=[0, 1, 0, 1])
    ax4.set_xlabel('x'); ax4.set_ylabel('y'); ax4.set_title('Ground Truth'); plt.colorbar(im, ax=ax4)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(eval_results['pred'].T, origin='lower', aspect='auto', cmap='viridis', extent=[0, 1, 0, 1])
    ax5.set_xlabel('x'); ax5.set_ylabel('y'); ax5.set_title('Prediction'); plt.colorbar(im, ax=ax5)
    
    ax6 = fig.add_subplot(gs[1, 2])
    error = (eval_results['pred'] - eval_results['gt'])
    im = ax6.imshow(np.abs(error).T, origin='lower', aspect='auto', cmap='Reds', extent=[0, 1, 0, 1])
    ax6.set_xlabel('x'); ax6.set_ylabel('y'); ax6.set_title(f'Error (max={np.abs(error).max():.2e})'); plt.colorbar(im, ax=ax6)
    
    if mode:
        fig.suptitle(f'DeepONet 2D - Mode ({mode[0]},{mode[1]})', fontsize=14, y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_deeponet_2d(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    def parse_mode(mode_str):
        return tuple(map(int, mode_str.split(",")))
    
    train_modes = [parse_mode(m) for m in args.train_modes]
    test_modes = [parse_mode(m) for m in args.test_modes]
    
    print(f"\n{'='*80}\nDeepONet 2D Training\n{'='*80}")
    print(f"Device: {device}\nTraining modes: {train_modes}\nTest modes: {test_modes}\n{'='*80}\n")

    dataset = SparseHeat2DDataset(10**6, args.M, args.nu, train_modes, args.Nc, args.Nbc, args.Nic, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = PIDeepONet2D(branch_dim=args.hidden_dim, trunk_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "step": [], "loss": [], "data": [], "pde": [], "bc": [], "ic": [], "lr": [],
        "eval_step": [], "eval_mse": [], "eval_mae": [], "eval_rel": [],
        "test_step": [], "test_mse": [], "test_mae": [], "test_rel": []
    }

    print("Starting training...\n" + "-" * 80)
    best_eval_mse = float("inf")
    data_iter = iter(loader)

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        stats = training_step(model, optimizer, batch, args.nu, args.weight_pde, args.weight_bc, args.weight_ic, device)
        scheduler.step()

        if step % args.print_every == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[{step:05d}] loss={stats['loss']:.3e} | data={stats['data']:.3e} pde={stats['pde']:.3e} "
                  f"bc={stats['bc']:.3e} ic={stats['ic']:.3e} | lr={current_lr:.2e}")
            history["step"].append(step); history["loss"].append(stats["loss"]); history["data"].append(stats["data"])
            history["pde"].append(stats["pde"]); history["bc"].append(stats["bc"]); history["ic"].append(stats["ic"])
            history["lr"].append(current_lr)

        if step % args.eval_every == 0 or step == args.steps:
            eval_results = evaluate_model_2d(model, args.nu, train_modes[0], device, M_eval=args.M)
            print(f"       EVAL (seen {train_modes[0]}) → MSE={eval_results['mse']:.3e} MAE={eval_results['mae']:.3e} RelErr={eval_results['rel_error']:.3%}")
            history["eval_step"].append(step); history["eval_mse"].append(eval_results["mse"])
            history["eval_mae"].append(eval_results["mae"]); history["eval_rel"].append(eval_results["rel_error"])
            
            test_results = evaluate_model_2d(model, args.nu, test_modes[0], device, M_eval=args.M)
            print(f"       TEST (unseen {test_modes[0]}) → MSE={test_results['mse']:.3e} MAE={test_results['mae']:.3e} RelErr={test_results['rel_error']:.3%}")
            history["test_step"].append(step); history["test_mse"].append(test_results["mse"])
            history["test_mae"].append(test_results["mae"]); history["test_rel"].append(test_results["rel_error"])

            if eval_results["mse"] < best_eval_mse:
                best_eval_mse = eval_results["mse"]
                print(f"       → New best MSE: {best_eval_mse:.3e}")
            print("-" * 80)

    print("\n" + "=" * 80 + "\nFINAL EVALUATION\n" + "=" * 80)
    print("\n--- Interpolation (Seen Modes) ---")
    for mode in train_modes:
        eval_seen = evaluate_model_2d(model, args.nu, mode, device, nx=128, ny=128, M_eval=args.M)
        print(f"Mode {mode}: MSE={eval_seen['mse']:.4e}, MAE={eval_seen['mae']:.4e}, RelErr={eval_seen['rel_error']:.4e}")
    
    print("\n--- Extrapolation (Unseen Modes) ---")
    for mode in test_modes:
        eval_unseen = evaluate_model_2d(model, args.nu, mode, device, nx=128, ny=128, M_eval=args.M)
        print(f"Mode {mode}: MSE={eval_unseen['mse']:.4e}, MAE={eval_unseen['mae']:.4e}, RelErr={eval_unseen['rel_error']:.4e}")
    
    print(f"\nBest training MSE: {best_eval_mse:.4e}\n" + "=" * 80)

    if args.plot:
        final_eval = evaluate_model_2d(model, args.nu, train_modes[0], device, nx=128, ny=128, M_eval=args.M)
        plot_results_2d(history, final_eval, "deeponet_2d_seen.png", train_modes[0])
        unseen_eval = evaluate_model_2d(model, args.nu, test_modes[0], device, nx=128, ny=128, M_eval=args.M)
        plot_results_2d(history, unseen_eval, "deeponet_2d_unseen.png", test_modes[0])

    print("\nTraining complete!\n" + "=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--train_modes", type=str, nargs="+", default=["1,1", "1,2"])
    ap.add_argument("--test_modes", type=str, nargs="+", default=["2,2", "2,3"])
    ap.add_argument("--Nc", type=int, default=4096)
    ap.add_argument("--Nbc", type=int, default=512)
    ap.add_argument("--Nic", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--weight_pde", type=float, default=0.1)
    ap.add_argument("--weight_bc", type=float, default=0.1)
    ap.add_argument("--weight_ic", type=float, default=0.1)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    train_deeponet_2d(args)


if __name__ == "__main__":
    main()
