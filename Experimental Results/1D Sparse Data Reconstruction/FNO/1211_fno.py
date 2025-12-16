#!/usr/bin/env python3
'''
training and testing loss calculation is fixed,

python 1211_fno.py --nu 0.1 --M 20 --steps 5000 --print_every 50 --eval_every 200

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

from neuralop.models import FNO


def exact_solution(x, t, nu=0.1, n=1):
    return torch.exp(-nu * (n * math.pi)**2 * t) * torch.sin(n * math.pi * x)


class FNOPointwise(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2 + hidden_dim, hidden_dim),
            nn.GELU(),
            *[layer for _ in range(num_layers-1) 
              for layer in (nn.Linear(hidden_dim, hidden_dim), nn.GELU())],
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, xt_sparse, u_sparse, xt_query):
        sparse_input = torch.cat([xt_sparse, u_sparse], dim=-1)
        encoded = self.encoder(sparse_input)
        
        context = encoded.mean(dim=1, keepdim=True)
        context = context.expand(-1, xt_query.shape[1], -1)
        
        query_input = torch.cat([xt_query, context], dim=-1)
        u_query = self.decoder(query_input)
        
        return u_query


class SparseHeatDataset(Dataset):
    def __init__(
        self,
        num_instances=10**6,
        M_sparse=200,
        nu=0.1,
        modes=[1, 2, 3],
        device='cpu'
    ):
        super().__init__()
        self.num_instances = num_instances
        self.M = M_sparse
        self.nu = nu
        self.modes = modes
        self.device = torch.device(device)
    
    def __len__(self):
        return self.num_instances
    
    def __getitem__(self, idx):
        mode_n = self.modes[idx % len(self.modes)]
        
        xt_sparse = torch.rand(self.M, 2, device=self.device)
        u_sparse = exact_solution(
            xt_sparse[:, 0], xt_sparse[:, 1], 
            self.nu, mode_n
        ).unsqueeze(-1)
        
        n_query = 1000
        xt_query = torch.rand(n_query, 2, device=self.device)
        u_query = exact_solution(
            xt_query[:, 0], xt_query[:, 1],
            self.nu, mode_n
        ).unsqueeze(-1)
        
        return {
            'xt_sparse': xt_sparse,
            'u_sparse': u_sparse,
            'xt_query': xt_query,
            'u_query': u_query,
            'mode': mode_n
        }


def train_step(model, optimizer, batch, device):
    model.train()
    optimizer.zero_grad()
    
    xt_sparse = batch['xt_sparse'].to(device)
    u_sparse = batch['u_sparse'].to(device)
    xt_query = batch['xt_query'].to(device)
    u_query = batch['u_query'].to(device)
    
    u_pred = model(xt_sparse, u_sparse, xt_query)
    
    loss = F.mse_loss(u_pred, u_query)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {'loss': loss.item()}


def evaluate_model(model, nu, mode_n, device, nx=256, nt=128, M_eval=200):
    model.eval()
    
    with torch.no_grad():
        xt_sparse = torch.rand(1, M_eval, 2, device=device)
        u_sparse = exact_solution(
            xt_sparse[0, :, 0], xt_sparse[0, :, 1],
            nu, mode_n
        ).unsqueeze(0).unsqueeze(-1)
        
        x = torch.linspace(0, 1, nx, device=device)
        t = torch.linspace(0, 1, nt, device=device)
        X, T = torch.meshgrid(x, t, indexing='ij')
        
        xt_eval = torch.stack([X, T], dim=-1).reshape(1, -1, 2)
        
        batch_size = 4096
        u_pred_list = []
        for i in range(0, xt_eval.shape[1], batch_size):
            xt_batch = xt_eval[:, i:i+batch_size]
            u_batch = model(xt_sparse, u_sparse, xt_batch)
            u_pred_list.append(u_batch)
        
        u_pred = torch.cat(u_pred_list, dim=1).reshape(nx, nt)
        u_true = exact_solution(X, T, nu, mode_n)
        
        mse = F.mse_loss(u_pred, u_true).item()
        mae = (u_pred - u_true).abs().mean().item()
        
        threshold = 1e-3
        mask = u_true.abs() > threshold
        if mask.sum() > 0:
            rel_error = ((u_pred[mask] - u_true[mask]).abs() / u_true[mask].abs()).mean().item()
        else:
            rel_error = float('nan')
        
        max_error = (u_pred - u_true).abs().max().item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rel_error': rel_error,
        'max_error': max_error,
        'prediction': u_pred.cpu(),
        'ground_truth': u_true.cpu(),
        'X': X.cpu(),
        'T': T.cpu()
    }


def plot_results(history, eval_results, save_path='fno_results.png'):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    if len(history['step']) > 0:
        ax1.semilogy(history['step'], history['loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
    
    # Train vs Test Error
    ax2 = fig.add_subplot(gs[0, 1])
    if len(history['eval_mse']) > 0:
        ax2.semilogy(history['eval_step'], history['eval_mse'], 'b-o', label='Train MSE', markersize=4)
        ax2.semilogy(history['eval_step'], history['eval_mae'], 'b--s', label='Train MAE', markersize=4)
        if len(history['test_mse']) > 0:
            ax2.semilogy(history['test_step'], history['test_mse'], 'r-o', label='Test MSE', markersize=4)
            ax2.semilogy(history['test_step'], history['test_mae'], 'r--s', label='Test MAE', markersize=4)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Error')
        ax2.set_title('Train vs Test Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Relative Error
    ax3 = fig.add_subplot(gs[0, 2])
    if len(history['eval_rel']) > 0:
        ax3.plot(history['eval_step'], history['eval_rel'], 'b-o', label='Train RelErr', markersize=4)
        if len(history['test_rel']) > 0:
            ax3.plot(history['test_step'], history['test_rel'], 'r-o', label='Test RelErr', markersize=4)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Relative Error')
        ax3.set_title('Relative Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Ground truth
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.contourf(eval_results['T'], eval_results['X'], 
                      eval_results['ground_truth'], levels=50, cmap='viridis')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Space x')
    ax4.set_title('Ground Truth')
    plt.colorbar(im, ax=ax4)
    
    # Prediction
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.contourf(eval_results['T'], eval_results['X'],
                      eval_results['prediction'], levels=50, cmap='viridis')
    ax5.set_xlabel('Time t')
    ax5.set_ylabel('Space x')
    ax5.set_title('FNO Prediction')
    plt.colorbar(im, ax=ax5)
    
    # Error
    ax6 = fig.add_subplot(gs[1, 2])
    error = (eval_results['prediction'] - eval_results['ground_truth']).abs()
    im = ax6.contourf(eval_results['T'], eval_results['X'], 
                      error, levels=50, cmap='Reds')
    ax6.set_xlabel('Time t')
    ax6.set_ylabel('Space x')
    ax6.set_title(f'Error (max={error.max():.2e})')
    plt.colorbar(im, ax=ax6)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to {save_path}")


def train_fno(args):
    device = torch.device(args.device)
    print(f"\n{'='*80}")
    print(f"FNO - Sparse Reconstruction Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Using: Official neuralop FNO")
    print(f"Sparse points (M): {args.M}")
    print(f"Training modes: {args.train_modes}")
    print(f"Test modes: {args.test_modes}")
    print(f"{'='*80}\n")
    
    dataset = SparseHeatDataset(
        M_sparse=args.M,
        nu=args.nu,
        modes=args.train_modes,
        device=device
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = FNOPointwise(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    
    # History with test tracking
    history = {
        'step': [], 'loss': [],
        'eval_step': [], 'eval_mse': [], 'eval_mae': [], 'eval_rel': [],
        'test_step': [], 'test_mse': [], 'test_mae': [], 'test_rel': []
    }
    
    print("Starting training...")
    print("-" * 80)
    
    best_mse = float('inf')
    data_iter = iter(loader)
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        stats = train_step(model, optimizer, batch, device)
        scheduler.step()
        
        if step % args.print_every == 0 or step == 1:
            print(f"[{step:05d}] loss={stats['loss']:.3e}")
            history['step'].append(step)
            history['loss'].append(stats['loss'])
        
        if step % args.eval_every == 0 or step == args.steps:
            eval_seen = evaluate_model(model, args.nu, args.train_modes[0], device, M_eval=args.M)
            print(f"       EVAL (seen mode {args.train_modes[0]}) → MSE={eval_seen['mse']:.3e} "
                  f"MAE={eval_seen['mae']:.3e}")
            
            history['eval_step'].append(step)
            history['eval_mse'].append(eval_seen['mse'])
            history['eval_mae'].append(eval_seen['mae'])
            history['eval_rel'].append(eval_seen['rel_error'])
            
            # Test set evaluation
            test_unseen = evaluate_model(model, args.nu, args.test_modes[0], device, M_eval=args.M)
            print(f"       TEST (unseen mode {args.test_modes[0]}) → MSE={test_unseen['mse']:.3e} "
                  f"MAE={test_unseen['mae']:.3e}")
            
            history['test_step'].append(step)
            history['test_mse'].append(test_unseen['mse'])
            history['test_mae'].append(test_unseen['mae'])
            history['test_rel'].append(test_unseen['rel_error'])
            
            if eval_seen['mse'] < best_mse:
                best_mse = eval_seen['mse']
            
            print("-" * 80)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    print("\n--- Interpolation (Seen Modes) ---")
    for mode in args.train_modes:
        eval_seen = evaluate_model(model, args.nu, mode, device, M_eval=args.M)
        print(f"Mode {mode}: MSE={eval_seen['mse']:.4e}, MAE={eval_seen['mae']:.4e}, "
              f"RelErr={eval_seen['rel_error']:.4e}")
    
    print("\n--- Extrapolation (Unseen Modes) ---")
    for mode in args.test_modes:
        eval_unseen = evaluate_model(model, args.nu, mode, device, M_eval=args.M)
        print(f"Mode {mode}: MSE={eval_unseen['mse']:.4e}, MAE={eval_unseen['mae']:.4e}, "
              f"RelErr={eval_unseen['rel_error']:.4e}")
    
    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"Best training MSE: {best_mse:.4e}")
    print("="*80)
    
    if args.plot:
        final_eval = evaluate_model(model, args.nu, args.train_modes[0], device, M_eval=args.M)
        plot_results(history, final_eval, 'fno_seen_mode.png')
        
        unseen_eval = evaluate_model(model, args.nu, args.test_modes[0], device, M_eval=args.M)
        plot_results(history, unseen_eval, 'fno_unseen_mode.png')
    
    return model, history, eval_seen, eval_unseen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--train_modes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--test_modes", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fno(args)