"""
PAT Model with 2D Heat Equation Support - FIXED VERSION

This version properly handles cross-attention between context and query positions.
Key fix: Separate handling of context (key/value) and query sequences.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class PATConfig:
    """Configuration for Physics-Aware Transformer."""
    
    def __init__(self):
        self.d_patch = 1  # Dimension of each patch (scalar field)
        self.d_pos = 2  # Dimension of position encoding (x, t) for 1D or (x, y, t) for 2D
        self.n_embd = 256  # Embedding dimension
        self.n_head = 8  # Number of attention heads
        self.n_layer = 4  # Number of transformer layers
        self.dropout = 0.1
        self.alpha = 1.0  # Physics guidance strength
        self.nu_bar = 0.1  # Reference diffusivity for heat kernel
        self.use_gradient_checkpointing = False


class PhysicsGuidedAttention(nn.Module):
    """
    Multi-head cross-attention with optional physics-guided bias.
    Properly handles different sequence lengths for context and query.
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.alpha = config.alpha
        self.nu_bar = config.nu_bar
        
        # Separate Q, K, V projections for cross-attention
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def compute_heat_kernel_bias(self, pos_ctx, pos_query):
        """
        Compute heat kernel bias for physics-guided attention.
        
        For 1D: pos shape [B, N, 2] where last dim is (x, t)
        For 2D: pos shape [B, N, 3] where last dim is (x, y, t)
        
        Returns bias [B, n_head, N_query, N_ctx]
        """
        B, N_ctx, D = pos_ctx.shape
        _, N_query, _ = pos_query.shape
        
        # Extract time coordinates
        t_ctx = pos_ctx[..., -1:]  # [B, N_ctx, 1]
        t_query = pos_query[..., -1:]  # [B, N_query, 1]
        
        # Time difference: Δt = t_query - t_ctx
        dt = t_query.unsqueeze(2) - t_ctx.unsqueeze(1)  # [B, N_query, N_ctx, 1]
        
        # Spatial difference
        if D == 2:  # 1D case: (x, t)
            x_ctx = pos_ctx[..., :1]  # [B, N_ctx, 1]
            x_query = pos_query[..., :1]  # [B, N_query, 1]
            dx = x_query.unsqueeze(2) - x_ctx.unsqueeze(1)  # [B, N_query, N_ctx, 1]
            spatial_dist_sq = dx ** 2  # [B, N_query, N_ctx, 1]
        
        elif D == 3:  # 2D case: (x, y, t)
            xy_ctx = pos_ctx[..., :2]  # [B, N_ctx, 2]
            xy_query = pos_query[..., :2]  # [B, N_query, 2]
            dxy = xy_query.unsqueeze(2) - xy_ctx.unsqueeze(1)  # [B, N_query, N_ctx, 2]
            spatial_dist_sq = (dxy ** 2).sum(dim=-1, keepdim=True)  # [B, N_query, N_ctx, 1]
        
        else:
            raise ValueError(f"Unsupported position dimension: {D}")
        
        # Heat kernel: exp(-|x_q - x_c|² / (4νΔt + ε))
        eps = 1e-6
        denominator = 4 * self.nu_bar * dt.abs() + eps
        kernel = torch.exp(-spatial_dist_sq / denominator)  # [B, N_query, N_ctx, 1]
        
        # Replicate across heads
        kernel = kernel.squeeze(-1)  # [B, N_query, N_ctx]
        bias = kernel.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # [B, n_head, N_query, N_ctx]
        
        return bias
    
    def forward(self, x_ctx, x_query, pos_ctx, pos_query):
        """
        Cross-attention forward pass.
        
        Args:
            x_ctx: context features [B, N_ctx, n_embd] (keys and values)
            x_query: query features [B, N_query, n_embd] (queries)
            pos_ctx: context positions [B, N_ctx, d_pos]
            pos_query: query positions [B, N_query, d_pos]
        
        Returns:
            output: [B, N_query, n_embd]
        """
        B, N_ctx, C = x_ctx.shape
        _, N_query, _ = x_query.shape
        
        # Compute Q from queries, K and V from context
        q = self.q_proj(x_query)  # [B, N_query, n_embd]
        k = self.k_proj(x_ctx)     # [B, N_ctx, n_embd]
        v = self.v_proj(x_ctx)     # [B, N_ctx, n_embd]
        
        # Reshape for multi-head attention
        q = q.view(B, N_query, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, N_query, head_dim]
        k = k.view(B, N_ctx, self.n_head, self.head_dim).transpose(1, 2)    # [B, n_head, N_ctx, head_dim]
        v = v.view(B, N_ctx, self.n_head, self.head_dim).transpose(1, 2)    # [B, n_head, N_ctx, head_dim]
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        # att shape: [B, n_head, N_query, N_ctx]
        
        # Add physics-guided bias if alpha > 0
        if self.alpha > 0:
            bias = self.compute_heat_kernel_bias(pos_ctx, pos_query)
            att = att + self.alpha * bias
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = att @ v  # [B, n_head, N_query, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, N_query, C)
        
        # Output projection
        y = self.proj(y)
        y = self.dropout(y)
        
        return y


class TransformerBlock(nn.Module):
    """Transformer block with cross-attention and feedforward."""
    
    def __init__(self, config):
        super().__init__()
        self.ln1_ctx = nn.LayerNorm(config.n_embd)
        self.ln1_query = nn.LayerNorm(config.n_embd)
        self.attn = PhysicsGuidedAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x_ctx, x_query, pos_ctx, pos_query):
        """
        Args:
            x_ctx: context features [B, N_ctx, n_embd]
            x_query: query features [B, N_query, n_embd]
            pos_ctx: context positions [B, N_ctx, d_pos]
            pos_query: query positions [B, N_query, d_pos]
        """
        # Cross-attention: queries attend to context
        x_query = x_query + self.attn(self.ln1_ctx(x_ctx), self.ln1_query(x_query), pos_ctx, pos_query)
        # Feedforward
        x_query = x_query + self.mlp(self.ln2(x_query))
        return x_query


class PATModel(nn.Module):
    """
    Physics-Aware Transformer for PDE solving.
    
    Supports both 1D and 2D heat equations:
    - 1D: ∂u/∂t = ν ∂²u/∂x²
    - 2D: ∂u/∂t = ν(∂²u/∂x² + ∂²u/∂y²)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.patch_embed = nn.Linear(config.d_patch, config.n_embd)
        self.pos_embed = nn.Sequential(
            nn.Linear(config.d_pos, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.d_patch)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, ctx_feats, ctx_pos, query_pos):
        """
        Forward pass for sparse reconstruction.
        
        Args:
            ctx_feats: context features [B, M, d_patch]
            ctx_pos: context positions [B, M, d_pos]
            query_pos: query positions [B, N, d_pos]
        
        Returns:
            predictions: [B, N, d_patch]
        """
        # Embed context
        x_ctx = self.patch_embed(ctx_feats) + self.pos_embed(ctx_pos)
        
        # Initialize query embeddings from positions only
        x_query = self.pos_embed(query_pos)
        
        # Process through transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x_query = torch.utils.checkpoint.checkpoint(
                    block, x_ctx, x_query, ctx_pos, query_pos
                )
            else:
                x_query = block(x_ctx, x_query, ctx_pos, query_pos)
        
        x_query = self.ln_f(x_query)
        
        # Predict at query positions
        preds = self.head(x_query)
        
        return preds
    
    def compute_pde_residual(self, u, pos, nu):
        """
        Compute PDE residual for 1D heat equation.
        PDE: ∂u/∂t - ν ∂²u/∂x² = 0
        
        Args:
            u: predictions [B, N, 1]
            pos: positions [B, N, 2] where pos[..., 0] is x, pos[..., 1] is t
            nu: diffusivity coefficient
        
        Returns:
            residual: [B, N, 1]
        """
        # Compute derivatives using autograd
        u_t = torch.autograd.grad(
            u, pos, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0][..., 1:2]  # ∂u/∂t
        
        u_x = torch.autograd.grad(
            u, pos, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0][..., 0:1]  # ∂u/∂x
        
        u_xx = torch.autograd.grad(
            u_x, pos, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][..., 0:1]  # ∂²u/∂x²
        
        residual = u_t - nu * u_xx
        
        return residual
    
    def compute_pde_residual_2d(self, u, pos, nu):
        """
        Compute PDE residual for 2D heat equation.
        PDE: ∂u/∂t - ν(∂²u/∂x² + ∂²u/∂y²) = 0
        
        Args:
            u: predictions [B, N, 1]
            pos: positions [B, N, 3] where pos[..., 0] is x, pos[..., 1] is y, pos[..., 2] is t
            nu: diffusivity coefficient
        
        Returns:
            residual: [B, N, 1]
        """
        # Compute time derivative
        u_t = torch.autograd.grad(
            u, pos, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0][..., 2:3]  # ∂u/∂t
        
        # Compute first spatial derivatives
        grads = torch.autograd.grad(
            u, pos, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = grads[..., 0:1]  # ∂u/∂x
        u_y = grads[..., 1:2]  # ∂u/∂y
        
        # Compute second spatial derivatives
        u_xx = torch.autograd.grad(
            u_x, pos, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][..., 0:1]  # ∂²u/∂x²
        
        u_yy = torch.autograd.grad(
            u_y, pos, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0][..., 1:2]  # ∂²u/∂y²
        
        # Heat equation residual
        residual = u_t - nu * (u_xx + u_yy)
        
        return residual
