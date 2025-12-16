#!/usr/bin/env python3
'''
python 1212_siren_2d.py \
    --M 100 \
    --nu 0.1 \
    --train_modes "1,1" "1,2" \
    --test_modes "2,2" "2,3" \
    --Nc 4000 \
    --Nbc 400 \
    --Nic 400 \
    --hidden_dim 256 \
    --num_layers 5 \
    --omega_0 30.0 \
    --omega_hidden 30.0 \
    --steps 30000 \
    --batch_size 4 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --lambda_pde 1.0 \
    --print_every 100 \
    --eval_every 500 \
    --plot
    
    
'''

import os
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def exact_solution_2d(x, y, t, nu=0.1, n=1, m=1):
    factor = nu * (math.pi ** 2) * (n**2 + m**2)
    return torch.exp(-factor * t) * torch.sin(n * math.pi * x) * torch.sin(m * math.pi * y)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN2D(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, omega_0=30.0, omega_hidden=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.first_layer = SineLayer(3, hidden_dim, is_first=True, omega_0=omega_0)
        self.hidden_layers = nn.ModuleList([
            SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_hidden)
            for _ in range(num_layers - 1)
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / omega_hidden
            self.final_layer.weight.uniform_(-bound, bound)
    
    def forward(self, xyt):
        h = self.first_layer(xyt)
        for layer in self.hidden_layers:
            h = layer(h)
        u = self.final_layer(h)
        return u


def compute_pde_residual_2d(model, xyt, nu):
    xyt.requires_grad_(True)
    u = model(xyt)
    grads = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = grads[..., 0:1]
    u_y = grads[..., 1:2]
    u_t = grads[..., 2:3]
    u_xx = torch.autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][..., 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][..., 1:2]
    residual = u_t - nu * (u_xx + u_yy)
    return residual


class SparseHeat2DDataset(Dataset):
    def __init__(self, num_instances=10**6, M_sparse=100, nu=0.1, modes=[(1,1),(1,2)], Nc=4000, Nbc=400, Nic=400, device='cpu'):
        super().__init__()
        self.num_instances = num_instances
        self.M = M_sparse
        self.nu = nu
        self.modes = modes
        self.device = torch.device(device)
        self.Nc = Nc
        self.Nbc = Nbc
        self.Nic = Nic
    
    def __len__(self):
        return self.num_instances
    
    def __getitem__(self, idx):
        n, m = self.modes[idx % len(self.modes)]
        xyt_sparse = torch.rand(self.M, 3, device=self.device)
        u_sparse = exact_solution_2d(xyt_sparse[:, 0], xyt_sparse[:, 1], xyt_sparse[:, 2], self.nu, n, m).unsqueeze(-1)
        xyt_colloc = torch.rand(self.Nc, 3, device=self.device)
        
        y_bc_a = torch.rand(self.Nbc, 1, device=self.device)
        t_bc_a = torch.rand(self.Nbc, 1, device=self.device)
        xyt_bc_a = torch.cat([torch.zeros_like(t_bc_a), y_bc_a, t_bc_a], dim=-1)
        u_bc_a = exact_solution_2d(xyt_bc_a[:, 0], xyt_bc_a[:, 1], xyt_bc_a[:, 2], self.nu, n, m).unsqueeze(-1)
        
        y_bc_b = torch.rand(self.Nbc, 1, device=self.device)
        t_bc_b = torch.rand(self.Nbc, 1, device=self.device)
        xyt_bc_b = torch.cat([torch.ones_like(t_bc_b), y_bc_b, t_bc_b], dim=-1)
        u_bc_b = exact_solution_2d(xyt_bc_b[:, 0], xyt_bc_b[:, 1], xyt_bc_b[:, 2], self.nu, n, m).unsqueeze(-1)
        
        x_bc_c = torch.rand(self.Nbc, 1, device=self.device)
        t_bc_c = torch.rand(self.Nbc, 1, device=self.device)
        xyt_bc_c = torch.cat([x_bc_c, torch.zeros_like(t_bc_c), t_bc_c], dim=-1)
        u_bc_c = exact_solution_2d(xyt_bc_c[:, 0], xyt_bc_c[:, 1], xyt_bc_c[:, 2], self.nu, n, m).unsqueeze(-1)
        
        x_bc_d = torch.rand(self.Nbc, 1, device=self.device)
        t_bc_d = torch.rand(self.Nbc, 1, device=self.device)
        xyt_bc_d = torch.cat([x_bc_d, torch.ones_like(t_bc_d), t_bc_d], dim=-1)
        u_bc_d = exact_solution_2d(xyt_bc_d[:, 0], xyt_bc_d[:, 1], xyt_bc_d[:, 2], self.nu, n, m).unsqueeze(-1)
        
        x_ic = torch.rand(self.Nic, 1, device=self.device)
        y_ic = torch.rand(self.Nic, 1, device=self.device)
        xyt_ic = torch.cat([x_ic, y_ic, torch.zeros_like(x_ic)], dim=-1)
        u_ic = exact_solution_2d(xyt_ic[:, 0], xyt_ic[:, 1], xyt_ic[:, 2], self.nu, n, m).unsqueeze(-1)
        
        return {
            'xyt_sparse': xyt_sparse, 'u_sparse': u_sparse, 'xyt_colloc': xyt_colloc,
            'xyt_bc_a': xyt_bc_a, 'u_bc_a': u_bc_a, 'xyt_bc_b': xyt_bc_b, 'u_bc_b': u_bc_b,
            'xyt_bc_c': xyt_bc_c, 'u_bc_c': u_bc_c, 'xyt_bc_d': xyt_bc_d, 'u_bc_d': u_bc_d,
            'xyt_ic': xyt_ic, 'u_ic': u_ic
        }


def train_step(model, optimizer, batch, nu, device, lambda_pde=1.0):
    model.train()
    optimizer.zero_grad()
    xyt_sparse = batch['xyt_sparse'].to(device)
    u_sparse = batch['u_sparse'].to(device)
    u_pred = model(xyt_sparse)
    loss_data = F.mse_loss(u_pred, u_sparse)
    
    xyt_colloc = batch['xyt_colloc'].to(device)
    residual = compute_pde_residual_2d(model, xyt_colloc, nu)
    loss_pde = (residual ** 2).mean()
    
    xyt_bc_a = batch['xyt_bc_a'].to(device)
    u_bc_a_pred = model(xyt_bc_a)
    loss_bc_a = F.mse_loss(u_bc_a_pred, batch['u_bc_a'].to(device))
    
    xyt_bc_b = batch['xyt_bc_b'].to(device)
    u_bc_b_pred = model(xyt_bc_b)
    loss_bc_b = F.mse_loss(u_bc_b_pred, batch['u_bc_b'].to(device))
    
    xyt_bc_c = batch['xyt_bc_c'].to(device)
    u_bc_c_pred = model(xyt_bc_c)
    loss_bc_c = F.mse_loss(u_bc_c_pred, batch['u_bc_c'].to(device))
    
    xyt_bc_d = batch['xyt_bc_d'].to(device)
    u_bc_d_pred = model(xyt_bc_d)
    loss_bc_d = F.mse_loss(u_bc_d_pred, batch['u_bc_d'].to(device))
    
    loss_bc = loss_bc_a + loss_bc_b + loss_bc_c + loss_bc_d
    
    xyt_ic = batch['xyt_ic'].to(device)
    u_ic_pred = model(xyt_ic)
    loss_ic = F.mse_loss(u_ic_pred, batch['u_ic'].to(device))
    
    loss = loss_data + lambda_pde * (loss_pde + loss_bc + loss_ic)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {'loss': loss.item(), 'data': loss_data.item(), 'pde': loss_pde.item(), 'bc': loss_bc.item(), 'ic': loss_ic.item()}


def evaluate_model_2d(model, nu, mode, device, nx=128, ny=128):
    model.eval()
    n, m = mode
    with torch.no_grad():
        t_eval = 0.5
        x = torch.linspace(0, 1, nx, device=device)
        y = torch.linspace(0, 1, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        T = torch.full_like(X, t_eval)
        xyt_eval = torch.stack([X, Y, T], dim=-1).reshape(-1, 3)
        batch_size = 8192
        u_pred_list = []
        for i in range(0, xyt_eval.shape[0], batch_size):
            xyt_batch = xyt_eval[i:i+batch_size]
            u_batch = model(xyt_batch)
            u_pred_list.append(u_batch)
        u_pred = torch.cat(u_pred_list, dim=0).reshape(nx, ny)
        u_true = exact_solution_2d(X, Y, T, nu, n, m)
        mse = F.mse_loss(u_pred, u_true).item()
        mae = (u_pred - u_true).abs().mean().item()
        threshold = 1e-3
        mask = u_true.abs() > threshold
        rel_error = ((u_pred[mask] - u_true[mask]).abs() / u_true[mask].abs()).mean().item() if mask.sum() > 0 else float('nan')
        max_error = (u_pred - u_true).abs().max().item()
    return {'mse': mse, 'mae': mae, 'rel_error': rel_error, 'max_error': max_error, 'prediction': u_pred.cpu(), 'ground_truth': u_true.cpu()}


def plot_results_2d(history, eval_results, save_path='siren_2d_results.png', mode=None):
    def to_numpy(data):
        if isinstance(data, list):
            return [x.item() if torch.is_tensor(x) else x for x in data]
        return data
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    if len(history['step']) > 0:
        ax1.semilogy(to_numpy(history['step']), to_numpy(history['loss']), 'b-', linewidth=2)
        ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(history['eval_mse']) > 0:
        ax2.semilogy(to_numpy(history['eval_step']), to_numpy(history['eval_mse']), 'b-o', label='Train MSE', markersize=4)
        if len(history['test_mse']) > 0:
            ax2.semilogy(to_numpy(history['test_step']), to_numpy(history['test_mse']), 'r-o', label='Test MSE', markersize=4)
        ax2.set_xlabel('Step'); ax2.set_ylabel('MSE'); ax2.set_title('MSE'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(history['eval_rel']) > 0:
        ax3.plot(to_numpy(history['eval_step']), [r*100 for r in to_numpy(history['eval_rel'])], 'b-o', label='Train', markersize=4)
        if len(history['test_rel']) > 0:
            ax3.plot(to_numpy(history['test_step']), [r*100 for r in to_numpy(history['test_rel'])], 'r-o', label='Test', markersize=4)
        ax3.set_xlabel('Step'); ax3.set_ylabel('Rel Error (%)'); ax3.set_title('Relative Error'); ax3.legend(); ax3.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(eval_results['ground_truth'].T, origin='lower', aspect='auto', cmap='viridis', extent=[0, 1, 0, 1])
    ax4.set_xlabel('x'); ax4.set_ylabel('y'); ax4.set_title('Ground Truth'); plt.colorbar(im, ax=ax4)
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(eval_results['prediction'].T, origin='lower', aspect='auto', cmap='viridis', extent=[0, 1, 0, 1])
    ax5.set_xlabel('x'); ax5.set_ylabel('y'); ax5.set_title('SIREN Prediction'); plt.colorbar(im, ax=ax5)
    ax6 = fig.add_subplot(gs[1, 2])
    error = (eval_results['prediction'] - eval_results['ground_truth']).abs()
    im = ax6.imshow(error.T, origin='lower', aspect='auto', cmap='Reds', extent=[0, 1, 0, 1])
    ax6.set_xlabel('x'); ax6.set_ylabel('y'); ax6.set_title(f'Error (max={error.max():.2e})'); plt.colorbar(im, ax=ax6)
    if mode:
        fig.suptitle(f'SIREN 2D - Mode ({mode[0]},{mode[1]})', fontsize=14, y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_siren_2d(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    def parse_mode(mode_str):
        return tuple(map(int, mode_str.split(",")))
    
    train_modes = [parse_mode(m) for m in args.train_modes]
    test_modes = [parse_mode(m) for m in args.test_modes]
    
    print(f"\n{'='*80}\nSIREN 2D Training\n{'='*80}")
    print(f"Device: {device}\nTraining modes: {train_modes}\nTest modes: {test_modes}\n{'='*80}\n")
    
    dataset = SparseHeat2DDataset(M_sparse=args.M, nu=args.nu, modes=train_modes, Nc=args.Nc, Nbc=args.Nbc, Nic=args.Nic, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SIREN2D(hidden_dim=args.hidden_dim, num_layers=args.num_layers, omega_0=args.omega_0, omega_hidden=args.omega_hidden).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)
    
    history = {'step': [], 'loss': [], 'data': [], 'pde': [], 'bc': [], 'ic': [], 'eval_step': [], 'eval_mse': [], 'eval_mae': [], 'eval_rel': [], 'test_step': [], 'test_mse': [], 'test_mae': [], 'test_rel': []}
    
    print("Starting training...\n" + "-" * 80)
    best_mse = float('inf')
    data_iter = iter(loader)
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        stats = train_step(model, optimizer, batch, args.nu, device, args.lambda_pde)
        scheduler.step()
        if step % args.print_every == 0 or step == 1:
            lr_current = scheduler.get_last_lr()[0]
            print(f"[{step:05d}] loss={stats['loss']:.3e} | data={stats['data']:.3e} pde={stats['pde']:.3e} bc={stats['bc']:.3e} ic={stats['ic']:.3e} | lr={lr_current:.2e}")
            history['step'].append(step); history['loss'].append(stats['loss']); history['data'].append(stats['data']); history['pde'].append(stats['pde']); history['bc'].append(stats['bc']); history['ic'].append(stats['ic'])
        if step % args.eval_every == 0 or step == args.steps:
            eval_results = evaluate_model_2d(model, args.nu, train_modes[0], device)
            print(f"       EVAL (seen {train_modes[0]}) → MSE={eval_results['mse']:.3e} MAE={eval_results['mae']:.3e} RelErr={eval_results['rel_error']:.2%}")
            history['eval_step'].append(step); history['eval_mse'].append(eval_results['mse']); history['eval_mae'].append(eval_results['mae']); history['eval_rel'].append(eval_results['rel_error'])
            test_results = evaluate_model_2d(model, args.nu, test_modes[0], device)
            print(f"       TEST (unseen {test_modes[0]}) → MSE={test_results['mse']:.3e} MAE={test_results['mae']:.3e} RelErr={test_results['rel_error']:.2%}")
            history['test_step'].append(step); history['test_mse'].append(test_results['mse']); history['test_mae'].append(test_results['mae']); history['test_rel'].append(test_results['rel_error'])
            if eval_results['mse'] < best_mse:
                best_mse = eval_results['mse']
                print(f"       → New best MSE: {best_mse:.3e}")
            print("-" * 80)
    
    elapsed = time.time() - start_time
    print("\n" + "="*80 + "\nFINAL EVALUATION\n" + "="*80)
    print("\n--- Interpolation (Seen Modes) ---")
    for mode in train_modes:
        eval_seen = evaluate_model_2d(model, args.nu, mode, device)
        print(f"Mode {mode}: MSE={eval_seen['mse']:.4e}, MAE={eval_seen['mae']:.4e}, RelErr={eval_seen['rel_error']:.4e}")
    print("\n--- Extrapolation (Unseen Modes) ---")
    for mode in test_modes:
        eval_unseen = evaluate_model_2d(model, args.nu, mode, device)
        print(f"Mode {mode}: MSE={eval_unseen['mse']:.4e}, MAE={eval_unseen['mae']:.4e}, RelErr={eval_unseen['rel_error']:.4e}")
    print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Best MSE: {best_mse:.4e}\n" + "="*80)
    
    if args.plot:
        final_eval = evaluate_model_2d(model, args.nu, train_modes[0], device)
        plot_results_2d(history, final_eval, 'siren_2d_seen.png', train_modes[0])
        unseen_eval = evaluate_model_2d(model, args.nu, test_modes[0], device)
        plot_results_2d(history, unseen_eval, 'siren_2d_unseen.png', test_modes[0])
    
    return model, history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--train_modes", nargs="+", type=str, default=["1,1", "1,2"])
    parser.add_argument("--test_modes", nargs="+", type=str, default=["2,2", "2,3"])
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--Nc", type=int, default=4000)
    parser.add_argument("--Nbc", type=int, default=400)
    parser.add_argument("--Nic", type=int, default=400)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--omega_0", type=float, default=30.0)
    parser.add_argument("--omega_hidden", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_pde", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_siren_2d(args)
