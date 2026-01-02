"""
Navier-Stokes PINNsformer Implementation
Physics-Informed Neural Network with Transformer architecture for solving Navier-Stokes equations
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import argparse
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import scipy.io

from util import *


# ==============================================================================
# Setup and Configuration
# ==============================================================================

def setup_environment(seed=0, device='cpu'):
    """
    Setup random seeds and device
    
    Args:
        seed: Random seed for reproducibility
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        device: PyTorch device object
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Handle device selection
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================

def load_navier_stokes_data(data_path='./cylinder_nektar_wake.mat'):
    """Load Navier-Stokes data from .mat file"""
    data = scipy.io.loadmat(data_path)
    
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']       # T x 1
    X_star = data['X_star']  # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data
    XX = np.tile(X_star[:,0:1], (1,T))  # N x T
    YY = np.tile(X_star[:,1:2], (1,T))  # N x T
    TT = np.tile(t_star, (1,N)).T       # N x T
    
    UU = U_star[:,0,:]  # N x T
    VV = U_star[:,1,:]  # N x T
    PP = P_star         # N x T
    
    x = XX.flatten()[:,None]  # NT x 1
    y = YY.flatten()[:,None]  # NT x 1
    t = TT.flatten()[:,None]  # NT x 1
    
    u = UU.flatten()[:,None]  # NT x 1
    v = VV.flatten()[:,None]  # NT x 1
    p = PP.flatten()[:,None]  # NT x 1
    
    return {
        'x': x, 'y': y, 't': t, 'u': u, 'v': v, 'p': p,
        'X_star': X_star, 'U_star': U_star, 'P_star': P_star,
        't_star': t_star, 'TT': TT, 'N': N, 'T': T
    }


def prepare_training_data(data, n_samples=2500, num_step=5, step=1e-2, device='cpu'):
    """Prepare training data with time sequences"""
    x, y, t, u, v = data['x'], data['y'], data['t'], data['u'], data['v']
    N, T = data['N'], data['T']
    
    # Random sampling
    idx = np.random.choice(N*T, n_samples, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    
    # Create time sequences
    x_train = np.expand_dims(np.tile(x_train[:], (num_step)), -1)
    y_train = np.expand_dims(np.tile(y_train[:], (num_step)), -1)
    t_train = make_time_sequence(t_train, num_step=num_step, step=step)
    
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
    t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
    u_train = torch.tensor(u_train, dtype=torch.float32, requires_grad=True).to(device)
    v_train = torch.tensor(v_train, dtype=torch.float32, requires_grad=True).to(device)
    
    return x_train, y_train, t_train, u_train, v_train


def prepare_test_data(data, snap=100, num_step=5, step=1e-2, device='cpu'):
    """Prepare test data for evaluation"""
    X_star = data['X_star']
    U_star = data['U_star']
    P_star = data['P_star']
    TT = data['TT']
    
    snap = np.array([snap])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    # Create time sequences
    x_star = np.expand_dims(np.tile(x_star[:], (num_step)), -1)
    y_star = np.expand_dims(np.tile(y_star[:], (num_step)), -1)
    t_star = make_time_sequence(t_star, num_step=num_step, step=step)
    
    # Convert to tensors
    x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(device)
    y_star = torch.tensor(y_star, dtype=torch.float32, requires_grad=True).to(device)
    t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True).to(device)
    
    return x_star, y_star, t_star, u_star, v_star, p_star


# ==============================================================================
# Model Architecture
# ==============================================================================

class WaveAct(nn.Module):
    """Wave activation function combining sine and cosine"""
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    """Feed-forward network with wave activation"""
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention"""
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        
    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention"""
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs): 
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    """Transformer encoder with multiple layers"""
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    """Transformer decoder with multiple layers"""
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class PINNsformer(nn.Module):
    """
    Physics-Informed Neural Network with Transformer architecture
    for solving Navier-Stokes equations
    """
    def __init__(self, d_out, d_model, d_hidden, N, heads):
        super(PINNsformer, self).__init__()
        
        self.linear_emb = nn.Linear(3, d_model)
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        ])

    def forward(self, x, y, t):
        src = torch.cat((x, y, t), dim=-1)
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        return output


def init_weights(m):
    """Initialize model weights using Xavier uniform"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# ==============================================================================
# Training
# ==============================================================================

def compute_pde_residuals(model, x_train, y_train, t_train, u_train, v_train):
    """
    Compute PDE residuals for Navier-Stokes equations
    
    Returns:
        loss: Total loss (data + physics)
        lossdata: Data loss only
    """
    # Forward pass
    psi_and_p = model(x_train, y_train, t_train)
    psi = psi_and_p[:,:,0:1]
    p = psi_and_p[:,:,1:2]
    
    # Compute velocity from stream function
    u = torch.autograd.grad(psi, y_train, grad_outputs=torch.ones_like(psi), 
                           retain_graph=True, create_graph=True)[0]
    v = -torch.autograd.grad(psi, x_train, grad_outputs=torch.ones_like(psi), 
                            retain_graph=True, create_graph=True)[0]
    
    # Compute derivatives for u
    u_t = torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(u), 
                             retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u), 
                             retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u), 
                             retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), 
                              retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_train, grad_outputs=torch.ones_like(u_y), 
                              retain_graph=True, create_graph=True)[0]
    
    # Compute derivatives for v
    v_t = torch.autograd.grad(v, t_train, grad_outputs=torch.ones_like(v), 
                             retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v), 
                             retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v), 
                             retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_train, grad_outputs=torch.ones_like(v_x), 
                              retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_train, grad_outputs=torch.ones_like(v_y), 
                              retain_graph=True, create_graph=True)[0]
    
    # Compute pressure derivatives
    p_x = torch.autograd.grad(p, x_train, grad_outputs=torch.ones_like(p), 
                             retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_train, grad_outputs=torch.ones_like(p), 
                             retain_graph=True, create_graph=True)[0]
    
    # Navier-Stokes residuals (Reynolds number = 100, nu = 0.01)
    f_u = u_t + (u*u_x + v*u_y) + p_x - 0.01*(u_xx + u_yy) 
    f_v = v_t + (u*v_x + v*v_y) + p_y - 0.01*(v_xx + v_yy)
    
    # Compute losses
    loss_data = torch.mean((u[:,0] - u_train)**2) + torch.mean((v[:,0] - v_train)**2)
    loss_physics = torch.mean(f_u**2) + torch.mean(f_v**2)
    loss = loss_data + loss_physics
    
    return loss, loss_data


def train_model(model, x_train, y_train, t_train, u_train, v_train, 
                optimizer, n_epochs=1000):
    """
    Train the PINNsformer model using LBFGS optimizer
    
    Returns:
        loss_track: List of total losses per epoch
        loss_data: List of data losses per epoch
    """
    loss_track = []
    loss_data_track = []
    
    for i in tqdm(range(n_epochs)):
        def closure():
            optimizer.zero_grad()
            loss, lossdata = compute_pde_residuals(
                model, x_train, y_train, t_train, u_train, v_train
            )
            loss_track.append(loss.item())
            loss_data_track.append(lossdata.item())
            loss.backward()
            return loss
        
        optimizer.step(closure)
    
    return loss_track, loss_data_track


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model, x_star, y_star, t_star, u_star, v_star, p_star):
    """
    Evaluate model on test data
    
    Returns:
        u_pred, v_pred, p_pred: Predicted values
        error_u, error_v, error_p: Relative L2 errors
    """
    # Forward pass
    psi_and_p = model(x_star, y_star, t_star)
    psi = psi_and_p[:,:,0:1]
    p_pred = psi_and_p[:,:,1:2]
    
    # Compute velocities
    u_pred = torch.autograd.grad(psi, y_star, grad_outputs=torch.ones_like(psi), 
                                 retain_graph=True, create_graph=True)[0]
    v_pred = -torch.autograd.grad(psi, x_star, grad_outputs=torch.ones_like(psi), 
                                  retain_graph=True, create_graph=True)[0]
    
    # Convert to numpy
    u_pred = u_pred.cpu().detach().numpy()[:,0]
    v_pred = v_pred.cpu().detach().numpy()[:,0]
    p_pred = p_pred.cpu().detach().numpy()[:,0]
    
    # Compute errors
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    
    return u_pred, v_pred, p_pred, error_u, error_v, error_p


# ==============================================================================
# Visualization
# ==============================================================================

def plot_results(p_star, p_pred, save_prefix='./ns_pinnsformer'):
    """Plot exact, predicted, and error for pressure field"""
    
    # Exact pressure
    plt.figure(figsize=(4,3))
    plt.imshow(p_star.reshape(50,100), extent=[-3,8,-2,2], aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Exact p(x,t)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_exact.png')
    plt.show()
    
    # Predicted pressure
    plt.figure(figsize=(4,3))
    plt.imshow(p_pred.reshape(50,100), extent=[-3,8,-2,2], aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted p(x,t)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_pred.png')
    plt.show()
    
    # Absolute error
    plt.figure(figsize=(4,3))
    plt.imshow(np.abs(p_pred - p_star).reshape(50,100), extent=[-3,8,-2,2], aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Absolute Error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_error.png')
    plt.show()


# ==============================================================================
# Main Execution
# ==============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train PINNsformer for Navier-Stokes equations'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cpu', 
        choices=['cpu', 'cuda'],
        help='Device to use for training (cpu or cuda)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='./cylinder_nektar_wake.mat',
        help='Path to data file'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=2500,
        help='Number of training samples'
    )
    
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=1000,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--d-model',
        type=int,
        default=32,
        help='Model dimension'
    )
    
    parser.add_argument(
        '--d-hidden',
        type=int,
        default=512,
        help='Hidden dimension'
    )
    
    parser.add_argument(
        '--n-layers',
        type=int,
        default=1,
        help='Number of transformer layers'
    )
    
    parser.add_argument(
        '--n-heads',
        type=int,
        default=2,
        help='Number of attention heads'
    )
    
    parser.add_argument(
        '--test-snap',
        type=int,
        default=100,
        help='Snapshot index for testing'
    )
    
    parser.add_argument(
        '--save-prefix',
        type=str,
        default='./ns_pinnsformer',
        help='Prefix for saving results'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_args()
    
    # Setup
    device = setup_environment(seed=args.seed, device=args.device)
    
    # Load data
    print("\nLoading data...")
    data = load_navier_stokes_data(args.data_path)
    
    # Prepare training data
    print("Preparing training data...")
    x_train, y_train, t_train, u_train, v_train = prepare_training_data(
        data, n_samples=args.n_samples, num_step=5, step=1e-2, device=device
    )
    
    # Create model
    print("\nCreating model...")
    model = PINNsformer(
        d_out=2, 
        d_hidden=args.d_hidden, 
        d_model=args.d_model, 
        N=args.n_layers, 
        heads=args.n_heads
    ).to(device)
    model.apply(init_weights)
    
    # Print model info
    n_params = get_n_params(model)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {n_params:,}")
    
    # Setup optimizer
    optimizer = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    
    # Train model
    print("\nTraining model...")
    loss_track, loss_data_track = train_model(
        model, x_train, y_train, t_train, u_train, v_train,
        optimizer, n_epochs=args.n_epochs
    )
    
    # Save results
    print("\nSaving training results...")
    np.save(f'{args.save_prefix}_loss.npy', loss_track)
    torch.save(model.state_dict(), f'{args.save_prefix}.pt')
    print(f"Final loss: {loss_track[-1]:.6e}")
    
    # Prepare test data
    print("\nPreparing test data...")
    x_star, y_star, t_star, u_star, v_star, p_star = prepare_test_data(
        data, snap=args.test_snap, num_step=5, step=1e-2, device=device
    )
    
    # Evaluate model
    print("Evaluating model...")
    u_pred, v_pred, p_pred, error_u, error_v, error_p = evaluate_model(
        model, x_star, y_star, t_star, u_star, v_star, p_star
    )
    
    # Print errors
    print(f"\nRelative L2 Errors:")
    print(f"  u-velocity: {error_u:.6e}")
    print(f"  v-velocity: {error_v:.6e}")
    print(f"  pressure:   {error_p:.6e}")
    
    # Compute L1 error for pressure
    error_p_l1 = np.linalg.norm(p_star - p_pred, 1) / np.linalg.norm(p_star, 1)
    print(f"  pressure (L1): {error_p_l1:.6e}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(p_star, p_pred, save_prefix=args.save_prefix)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
