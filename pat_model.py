import math
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# -----------------------------
# NanoGPT utilities (kept)
# -----------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (same as NanoGPT)."""
    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

# -----------------------------
# [PAT] Physics-aware attention
# -----------------------------

class PhysSelfAttention(nn.Module):
    """
    Multi-head self-attention with additive physics bias Gamma.
    CUDA-OPTIMIZED: Pre-allocated buffers, no runtime tensor creation.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        
        # CUDA FIX: Pre-allocate mask value buffer
        self.register_buffer('neg_inf_value', torch.tensor(-1e9))

    def forward(self, x, gamma_bias: Optional[torch.Tensor] = None):
        """
        x: (B, P, C) context tokens
        gamma_bias: (B or 1, 1, P, P) additive bias (heat-kernel log weights)
        """
        B, P, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)  # (B, H, P, Hd)
        k = k.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash and gamma_bias is None:
            # fast path (no external bias)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
        else:
            # manual attention to incorporate gamma_bias
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B,H,P,P)
            if gamma_bias is not None:
                # broadcast to (B, H, P, P) if needed
                if gamma_bias.dim() == 4:
                    gb = gamma_bias
                    if gb.size(0) == 1:  # (1,1,P,P)
                        gb = gb.expand(B, 1, P, P)
                    if gb.size(1) == 1:
                        gb = gb.expand(B, self.n_head, P, P)
                    att = att + gb
                else:
                    raise ValueError("gamma_bias must have shape (B or 1, 1, P, P)")
            
            # CUDA FIX: Use pre-allocated buffer, avoid device sync
            inf_mask = torch.isinf(att) & (att < 0)
            if inf_mask.any():
                att = torch.where(inf_mask, self.neg_inf_value, att)
            att = torch.clamp(att, min=-1e9, max=1e9)
            
            att = F.softmax(att, dim=-1)
            
            # CUDA FIX: Only check NaN in debug mode (remove from production)
            # if torch.isnan(att).any():
            #     att = torch.where(torch.isnan(att), 
            #                      torch.ones_like(att) / att.size(-1), 
            #                      att)
            
            att = self.attn_dropout(att)
            y = att @ v  # (B,H,P,Hd)

        y = y.transpose(1, 2).contiguous().view(B, P, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Same as NanoGPT MLP."""
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class PATBlock(nn.Module):
    """
    Transformer block using PhysSelfAttention.
    CUDA-OPTIMIZED: Added gradient checkpointing support.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool, use_checkpoint: bool = False):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = PhysSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp  = MLP(n_embd, dropout, bias)
        self.use_checkpoint = use_checkpoint
        
    def _forward_impl(self, x, gamma_bias):
        x = x + self.attn(self.ln_1(x), gamma_bias=gamma_bias)
        x = x + self.mlp(self.ln_2(x))
        return x
    
    def forward(self, x, gamma_bias=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, gamma_bias, use_reentrant=False)
        else:
            return self._forward_impl(x, gamma_bias)

# -----------------------------
# [PAT] Cross-attention for queries
# -----------------------------

class CrossAttention(nn.Module):
    """
    Query (from coordinate encoding) attends to context tokens C.
    Returns query-conditioned context vector g(x,t).
    """
    def __init__(self, q_dim: int, ctx_dim: int, out_dim: int, n_head: int, bias: bool, dropout: float):
        super().__init__()
        assert out_dim % n_head == 0
        self.n_head = n_head
        self.dk = out_dim // n_head
        self.q_proj = nn.Linear(q_dim, out_dim, bias=bias)
        self.k_proj = nn.Linear(ctx_dim, out_dim, bias=bias)
        self.v_proj = nn.Linear(ctx_dim, out_dim, bias=bias)
        self.o_proj = nn.Linear(out_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.ln_out = LayerNorm(out_dim, bias=bias)

    def forward(self, q, C):
        """
        q: (B, Nq, q_dim) - typically Nq = number of query points in minibatch
        C: (B, P, ctx_dim) - context tokens
        returns g: (B, Nq, out_dim)
        """
        B, Nq, _ = q.shape
        _, P, _  = C.shape
        Q = self.q_proj(q).view(B, Nq, self.n_head, self.dk).transpose(1, 2)  # (B,H,Nq,dk)
        K = self.k_proj(C).view(B, P,  self.n_head, self.dk).transpose(1, 2)  # (B,H,P, dk)
        V = self.v_proj(C).view(B, P,  self.n_head, self.dk).transpose(1, 2)

        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))  # (B,H,Nq,P)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        Y = att @ V  # (B,H,Nq,dk)
        Y = Y.transpose(1, 2).contiguous().view(B, Nq, self.n_head * self.dk)  # (B,Nq,out_dim)
        Y = self.o_proj(Y)
        return self.ln_out(Y)

# -----------------------------
# [PAT] SIREN + FiLM + freq gate
# -----------------------------

class Sine(nn.Module):
    def forward(self, x): return torch.sin(x)

class SIRENLayer(nn.Module):
    """
    Standard SIREN linear + sine with configurable omega0.
    """
    def __init__(self, in_dim, out_dim, omega0=30.0, is_first=False, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.omega0 = omega0
        self.is_first = is_first
        # SIREN init
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_dim, 1/in_dim)
            else:
                self.linear.weight.uniform_(-math.sqrt(6/in_dim)/omega0, math.sqrt(6/in_dim)/omega0)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))

class FiLMHyper(nn.Module):
    """
    Produces FiLM (gamma, beta) and frequency gates per INR layer from g(x,t) and global token.
    """
    def __init__(self, in_dim, layer_hidden: int, n_layers: int, width: int):
        super().__init__()
        self.n_layers = n_layers
        self.width = width
        out_dim = n_layers * (2 * width + width)  # gamma(width) + beta(width) + omega(width)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, layer_hidden), nn.GELU(),
            nn.Linear(layer_hidden, layer_hidden), nn.GELU(),
            nn.Linear(layer_hidden, out_dim)
        )

    def forward(self, g):
        """
        g: (B, Nq, in_dim)
        returns dict of lists: gammas, betas, omegas (each list length n_layers, tensors (B,Nq,width))
        """
        B, Nq, _ = g.shape
        vec = self.mlp(g)  # (B,Nq,out_dim)
        # split into per-layer chunks
        chunks = torch.chunk(vec, self.n_layers, dim=-1)
        gammas, betas, omegas = [], [], []
        for ch in chunks:
            gw, bw, ow = torch.split(ch, [self.width, self.width, self.width], dim=-1)
            gammas.append(gw)
            betas.append(bw)
            # constrain freq positives via softplus + small floor
            omegas.append(F.softplus(ow) + 1e-2)
        return gammas, betas, omegas

class FiLMSIREN(nn.Module):
    """
    FiLM + adaptive-frequency gated SIREN.
    """
    def __init__(self, in_dim: int, width: int, depth: int, omega0: float, hyper_in_dim: int, hyper_hidden: int):
        super().__init__()
        self.depth = depth
        self.width = width
        self.first = SIRENLayer(in_dim, width, omega0=omega0, is_first=True)
        self.hidden = nn.ModuleList([nn.Linear(width, width, bias=True) for _ in range(depth-1)])
        self.sine = Sine()
        self.out = nn.Linear(width, 1, bias=True)
        self.hyper = FiLMHyper(hyper_in_dim, hyper_hidden, n_layers=depth-1, width=width)

    def forward(self, phi, g, cglob):
        """
        phi: (B,Nq,in_dim) Fourier features of (x,t)
        g:   (B,Nq,hyper_in_dim') cross-attended context
        cglob: (B,1,ctx_dim) global token broadcastable
        returns u: (B,Nq,1)
        """
        B, Nq, _ = phi.shape
        # concat g and global
        if cglob is not None:
            cgb = cglob.expand(-1, Nq, -1)
            hyper_in = torch.cat([g, cgb], dim=-1)  # (B,Nq,hyper_in_dim)
        else:
            hyper_in = g

        # predict FiLM params and per-layer omegas for hidden layers only
        gammas, betas, omegas = self.hyper(hyper_in)

        # first layer (fixed omega0 SIREN)
        h = self.first(phi)  # (B,Nq,W)

        # FiLM-modulated hidden layers
        for i, lin in enumerate(self.hidden):
            z = lin(h)  # (B,Nq,W)
            z = gammas[i] * z + betas[i]         # FiLM
            z = omegas[i] * z                    # frequency gating
            h = self.sine(z)

        u = self.out(h)
        return u

# -----------------------------
# [PAT] Fourier features
# -----------------------------

class FourierFeatures(nn.Module):
    """phi(x,t) = [x,t,sin(2pi B[x,t]), cos(2pi B[x,t])]."""
    def __init__(self, in_dim=2, m=64, logspace=True):
        super().__init__()
        self.in_dim = in_dim
        self.m = m
        # fixed B
        if logspace:
            # log-spaced frequencies
            exps = torch.linspace(0, 1, steps=m)
            freqs = (2.0 ** (10.0 * exps)).unsqueeze(0)  # (1,m)
        else:
            freqs = torch.linspace(1.0, 2**10, steps=m).unsqueeze(0)
        # project [x,t] by a diagonal frequency bank (simple variant)
        self.register_buffer("B", torch.cat([freqs, freqs], dim=0))  # (2,m)

    def forward(self, xt):  # xt: (B,Nq,2) with columns [x, t]
        B, Nq, _ = xt.shape
        # linear map: (B,Nq,2) @ (2,m) -> (B,Nq,m)
        proj = torch.matmul(xt, self.B)              # (B,Nq,m)
        s = torch.sin(2*math.pi*proj)
        c = torch.cos(2*math.pi*proj)
        return torch.cat([xt, s, c], dim=-1)         # (B,Nq,2 + m + m)

# -----------------------------
# [PAT] Config
# -----------------------------

@dataclass
class PATConfig:
    # Transformer/context
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

    # Patch feature dim -> embedding
    d_patch: int = 32

    # Cross-attn dims
    d_query: int = 2 + 128 + 128  # [x,t] + sin/cos(m=128)
    d_cross: int = 256

    # INR
    siren_width: int = 256
    siren_depth: int = 5
    siren_omega0: float = 30.0
    hyper_hidden: int = 256

    # Fourier features
    m_ff: int = 128

    # Loss weights
    use_uncertainty: bool = True
    lambda_spec: float = 0.0
    lambda_reg: float = 0.0
    
    # Domain bounds for boundary conditions
    x_min: float = 0.0
    x_max: float = 1.0
    
    # CUDA optimization
    use_gradient_checkpointing: bool = True

# -----------------------------
# [PAT] Full Model
# -----------------------------

class PATModel(nn.Module):
    def __init__(self, cfg: PATConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.d_patch, cfg.n_embd, bias=cfg.bias)
        self.pos_enc = nn.Linear(2, cfg.n_embd, bias=False)    # simple linear pos-proj

        # Transformer blocks with gradient checkpointing
        self.blocks = nn.ModuleList([
            PATBlock(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.bias, 
                    use_checkpoint=cfg.use_gradient_checkpointing)
            for _ in range(cfg.n_layer)
        ])
        self.ln_ctx = LayerNorm(cfg.n_embd, bias=cfg.bias)

        # CLS/global token
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # Cross-attention query -> context
        self.cross = CrossAttention(
            q_dim=(2 + 2*cfg.m_ff), ctx_dim=cfg.n_embd, out_dim=cfg.d_cross,
            n_head=cfg.n_head, bias=cfg.bias, dropout=cfg.dropout
        )

        # Fourier features for (x,t)
        self.ff = FourierFeatures(in_dim=2, m=cfg.m_ff, logspace=True)

        # FiLM SIREN: hyper input = [g(x,t), c_global]
        hyper_in_dim = cfg.d_cross + cfg.n_embd
        self.inr = FiLMSIREN(
            in_dim=(2 + 2*cfg.m_ff), width=cfg.siren_width, depth=cfg.siren_depth,
            omega0=cfg.siren_omega0, hyper_in_dim=hyper_in_dim, hyper_hidden=cfg.hyper_hidden
        )

        # Uncertainty weights (optional)
        if cfg.use_uncertainty:
            # FIXED: Initialize to log(1.0) = 0 for sigma=1.0, not log(0)
            # This makes initial weighting neutral (not aggressive)
            self.logsig_data = nn.Parameter(torch.zeros(1))  # sigma = 1.0
            self.logsig_pde  = nn.Parameter(torch.zeros(1))  # sigma = 1.0
            self.logsig_bc   = nn.Parameter(torch.zeros(1))  # sigma = 1.0
            self.logsig_ic   = nn.Parameter(torch.zeros(1))  # sigma = 1.0
        
        # CUDA FIX: Pre-allocate buffer for -inf masks
        self.register_buffer('neg_inf', torch.tensor(float('-inf')))

    # --------- physics bias Gamma (heat kernel) ----------
    def heat_kernel_bias(self, ctx_pos: torch.Tensor, alpha: float, nu_bar: float) -> torch.Tensor:
        """
        ctx_pos: (B,P,2) with (x_p, t_p)
        returns Gamma: (B,1,P,P) with log heat-kernel, -inf on invalid (Delta t<=0)
        CUDA-OPTIMIZED: Better numerical stability, pre-allocated buffers.
        """
        B, P, _ = ctx_pos.shape
        x = ctx_pos[..., 0].unsqueeze(-1)  # (B,P,1)
        t = ctx_pos[..., 1].unsqueeze(-1)  # (B,P,1)

        dx2 = (x - x.transpose(1, 2)) ** 2  # (B,P,P)
        dt  = (t - t.transpose(1, 2))       # (B,P,P)
        
        mask = torch.ones_like(dt, dtype=torch.bool)  # Allow all connections
        # mask = (dt > 0).float()
        
        # CUDA FIX: Better numerical stability
        safe_dt = torch.clamp(dt, min=1e-6)
        safe_dx2 = torch.clamp(dx2, min=1e-10)
        
        # Compute log term with numerical stability
        log_term = -0.5 * torch.log(torch.clamp(4*math.pi*nu_bar*safe_dt, min=1e-10))
        exp_term = -safe_dx2 / torch.clamp(4*nu_bar*safe_dt, min=1e-10)
        exp_term = torch.clamp(exp_term, min=-50, max=50)
        
        logG = log_term + exp_term
        logG = torch.clamp(logG, min=-50, max=50)
        
        # CUDA FIX: Use pre-allocated buffer
        logG = torch.where(mask > 0, alpha * logG, self.neg_inf)
        Gamma = logG.unsqueeze(1)     # (B,1,P,P)
        return Gamma

    # --------- forward passes ----------
    def encode_context(self, ctx_feats: torch.Tensor, ctx_pos: torch.Tensor,
                       alpha: float = 1.0, nu_bar: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds physics-aware context tokens.
        CUDA-OPTIMIZED: Pre-allocated buffers for padding.
        """
        B, P, _ = ctx_feats.shape
        
        e = self.patch_embed(ctx_feats) + self.pos_enc(ctx_pos)  # (B,P,C)
        cls = self.cls.expand(B, -1, -1)                        # (B,1,C)
        C = torch.cat([cls, e], dim=1)                          # (B,P+1,C)

        # Physics bias computed only across patch tokens (exclude cls).
        gamma = self.heat_kernel_bias(ctx_pos, alpha, nu_bar)   # (B,1,P,P)
        
        # CUDA FIX: More efficient padding with pre-allocated buffer
        pad_row = self.neg_inf.expand(B, 1, 1, P)
        gamma_full = torch.cat([pad_row, gamma], dim=2)
        pad_col = self.neg_inf.expand(B, 1, P+1, 1)
        gamma_full = torch.cat([pad_col, gamma_full], dim=3)

        for blk in self.blocks:
            C = blk(C, gamma_bias=gamma_full)

        C = self.ln_ctx(C)
        cglob = C[:, :1, :]             # (B,1,C)
        C_ctx = C[:, 1:, :]             # (B,P,C)
        return C_ctx, cglob

    def predict_u(self, C: torch.Tensor, cglob: torch.Tensor, xt_q: torch.Tensor) -> torch.Tensor:
        """
        Computes u(x,t) for query coordinates.
        C should be patch tokens only (no CLS).
        """
        phi = self.ff(xt_q)                    # (B,Nq,2+2m)
        g   = self.cross(phi, C)               # (B,Nq,d_cross)
        u   = self.inr(phi, g, cglob)          # (B,Nq,1)
        return u

    # --------- loss helpers ----------
    def _diffusion_residual(self, u, xt, f, nu):
        """
        u: (B,N,1) predictions at xt
        xt: (B,N,2) coords (must already have requires_grad=True)
        f: (B,N,1) forcing (zeros if None)
        nu: float or (B,1,1)
        returns r = u_t - nu * u_xx - f  -> (B,N,1)
        CUDA-OPTIMIZED: Better error handling removed print statements.
        """
        if f is None: 
            f = torch.zeros_like(u)

        # Ensure xt requires grad
        if not xt.requires_grad:
            xt = xt.requires_grad_(True)
        
        # du/dt
        dudt = torch.autograd.grad(
            u, xt, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True, allow_unused=False
        )[0][..., 1:2]

        # du/dx
        dudx = torch.autograd.grad(
            u, xt, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True, allow_unused=False
        )[0][..., 0:1]

        # d2u/dx2
        d2udx2 = torch.autograd.grad(
            dudx, xt, grad_outputs=torch.ones_like(dudx),
            retain_graph=True, create_graph=True, allow_unused=False
        )[0][..., 0:1]

        if isinstance(nu, float):
            nu_t = nu
        else:
            nu_t = nu

        r = dudt - nu_t * d2udx2 - f
        return r

    def _uncert_weight(self, loss, logsig):
        """
        Uncertainty weighting with proper formulation.
        Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al. 2018)
        
        The correct formula is:
        L = (1 / (2 * sigma^2)) * task_loss + log(sigma)
        
        With sigma = exp(logsig):
        L = 0.5 * exp(-2*logsig) * task_loss + logsig
        
        PROBLEM: When task_loss is very small (like pde=6e-8) and logsig is negative,
        the +logsig term dominates and can make the total contribution negative!
        
        SOLUTION: Initialize logsig to reasonable values and clamp tightly, OR
        use the alternative formulation that's always positive.
        """
        # FIXED: Tighter clamping to prevent pathological behavior
        # Keep logsig in [-2, 2] so sigma in [0.135, 7.39]
        logsig_clamped = torch.clamp(logsig, -2, 2)
        
        weighted_loss = 0.5 * torch.exp(-2*logsig_clamped) * loss
        regularizer = logsig_clamped
        
        total = weighted_loss + regularizer
        
        # SAFETY: If somehow still negative, fall back to unweighted loss
        if total < 0:
            return loss
        
        return total

    # --------- public forward ----------
    def forward(self,
                ctx_feats: torch.Tensor,   # (B,P,d_patch)
                ctx_pos:   torch.Tensor,   # (B,P,2)  normalized [x,t] of patch centers
                xt_q:      torch.Tensor,   # (B,Nq,2) query coords
                f_q: Optional[torch.Tensor] = None,  # (B,Nq,1)
                training_dict: Optional[Dict] = None,
                pde_weight=0.0):
        """
        If training_dict is None: returns {"u": u(x,t)}.
        Else: computes and returns loss dict with components.
        """
        # 1) Encode context with physics-aware bias
        C, cglob = self.encode_context(ctx_feats, ctx_pos)

        # 2) Predict u at query points
        if training_dict is not None:
            xt_q = xt_q.requires_grad_(True)
        u_q  = self.predict_u(C, cglob, xt_q)

        if training_dict is None:
            return {"u": u_q}

        # ----------------- Losses -----------------
        B = xt_q.size(0)
        device = xt_q.device
        loss_data = torch.tensor(0.0, device=device)
        loss_pde  = torch.tensor(0.0, device=device)
        loss_bc   = torch.tensor(0.0, device=device)
        loss_ic   = torch.tensor(0.0, device=device)

        # nu
        nu = training_dict.get("nu", 0.1)

        # (a) HR data loss
        if "hr" in training_dict and training_dict["hr"] is not None:
            xh = training_dict["hr"]["x"]   # (B,Nh,2)
            yh = training_dict["hr"]["y"]   # (B,Nh,1)
            uh = self.predict_u(C, cglob, xh)
            loss_hr = F.mse_loss(uh, yh)
        else:
            loss_hr = torch.tensor(0.0, device=device)

        # (b) LR data loss via batched quadrature points
        if "lr" in training_dict and training_dict["lr"] is not None:
            ylr = training_dict["lr"]["y_lr"]  # (B,Nl,1)
            if "quad_points" in training_dict["lr"]:
                quad_pts = training_dict["lr"]["quad_points"]
                B_lr, Nl, Nqp, _ = quad_pts.shape
                quad_flat = quad_pts.view(B_lr, Nl*Nqp, 2)
                u_quad = self.predict_u(C, cglob, quad_flat)
                u_quad = u_quad.view(B_lr, Nl, Nqp, 1)
                ulr = u_quad.mean(dim=2)
                loss_lr = F.mse_loss(ulr, ylr)
            else:
                loss_lr = torch.tensor(0.0, device=device)
        else:
            loss_lr = torch.tensor(0.0, device=device)
        loss_data = loss_hr + loss_lr

        # (c) PDE residual loss at interior collocation points
        if "pde" in training_dict and training_dict["pde"] is not None:
            xr = training_dict["pde"]["colloc"]   # (B,Nr,2)
            fr = training_dict["pde"].get("f_r", None)  # optional forcing at collocation
            # FIXED: Ensure requires_grad for derivative computation
            if not xr.requires_grad:
                xr = xr.clone().requires_grad_(True)
            ur = self.predict_u(C, cglob, xr)
            rr = self._diffusion_residual(ur, xr, fr, nu)
            # loss_pde = (rr**2).mean()
            # loss_pde = F.smooth_l1_loss(rr, torch.zeros_like(rr))
            loss_pde = torch.log(1 + (rr**2).mean())

            # loss_pde = ((rr / (ur.abs() + 1e-6))**2).mean()
            # loss_pde = pde_weight*loss_pde

        # (d) Boundary conditions (Dirichlet example)
        # FIXED: Use configurable domain bounds instead of hardcoding
        if "bc" in training_dict and training_dict["bc"] is not None:
            bc = training_dict["bc"]
            x_min = self.cfg.x_min
            x_max = self.cfg.x_max
            
            if "a_times" in bc and "g_a" in bc:
                ta = bc["a_times"]  # (B,Na,1)
                xa = torch.cat([torch.full_like(ta, x_min), ta], dim=-1)  # x=x_min
                ua = self.predict_u(C, cglob, xa)
                loss_bc_a = F.mse_loss(ua, bc["g_a"])
            else:
                loss_bc_a = torch.tensor(0.0, device=device)

            if "b_times" in bc and "g_b" in bc:
                tb = bc["b_times"]  # (B,Nb,1)
                xb = torch.cat([torch.full_like(tb, x_max), tb], dim=-1)   # x=x_max
                ub = self.predict_u(C, cglob, xb)
                loss_bc_b = F.mse_loss(ub, bc["g_b"])
            else:
                loss_bc_b = torch.tensor(0.0, device=device)

            loss_bc = loss_bc_a + loss_bc_b
            # loss_bc = pde_weight*loss_bc

        # (e) Initial condition (Dirichlet at t=0)
        if "ic" in training_dict and training_dict["ic"] is not None:
            x0 = training_dict["ic"]["x0"]               # (B,N0,1)
            u0 = training_dict["ic"]["u0"]               # (B,N0,1)
            xt0 = torch.cat([x0, torch.zeros_like(x0)], dim=-1)  # t=0
            u_hat0 = self.predict_u(C, cglob, xt0)
            loss_ic = F.mse_loss(u_hat0, u0)
            # loss_ic =  pde_weight*loss_ic

        # (f) Uncertainty-weighted sum
        if self.cfg.use_uncertainty:
            loss = self._uncert_weight(loss_data, self.logsig_data) \
                 + pde_weight*(self._uncert_weight(loss_pde,  self.logsig_pde)  \
                 + pde_weight*self._uncert_weight(loss_bc,   self.logsig_bc)   \
                 + pde_weight*self._uncert_weight(loss_ic,   self.logsig_ic))
            # print(f'pde:',{self._uncert_weight(loss_pde,  self.logsig_pde)})
            # print(f'simle pde: ', loss_pde)
            # print(f'bc:', {self._uncert_weight(loss_bc,   self.logsig_bc)})
            # print(f'ic:', {self._uncert_weight(loss_ic,   self.logsig_ic)})

        else:
            loss = loss_data + loss_pde + loss_bc + loss_ic

            
        # loss = loss_data + loss_bc
        return {
            "u": u_q,
            "loss": loss,
            "loss_components": {
                "data": loss_data.detach(),
                "pde":  loss_pde.detach(),
                "bc":   loss_bc.detach(),
                "ic":   loss_ic.detach()
            }
        }

    # ----------------- optimizer config (kept style) -----------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() <  2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                      **({'fused': True} if use_fused else {}))
        print(f"using fused AdamW: {use_fused}")
        return optimizer    