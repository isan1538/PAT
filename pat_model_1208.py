# addressing the loss calculation inconsistency, now error is calculated only in training code (pat_training_1208.py)

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class PATConfig:
    d_patch: int = 4
    d_pos: int = 2
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    bias: bool = True
    x_min: float = 0.0
    x_max: float = 1.0
    use_gradient_checkpointing: bool = False
    alpha: float = 1.0
    nu_bar: float = 0.1


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class PhysSelfAttention(nn.Module):
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
        self.register_buffer("neg_inf_value", torch.tensor(-1e9))

    def forward(self, x, gamma_bias: Optional[torch.Tensor] = None):
        B, P, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash and gamma_bias is None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if gamma_bias is not None:
                if gamma_bias.dim() == 4:
                    gb = gamma_bias
                    if gb.size(0) == 1:
                        gb = gb.expand(B, 1, P, P)
                    if gb.size(1) == 1:
                        gb = gb.expand(B, self.n_head, P, P)
                    att = att + gb
                else:
                    raise ValueError(
                        "gamma_bias must have shape (B or 1, 1, P, P)"
                    )

            inf_mask = torch.isinf(att) & (att < 0)
            if inf_mask.any():
                att = torch.where(inf_mask, self.neg_inf_value, att)
            att = torch.clamp(att, min=-1e9, max=1e9)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, P, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class PATBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = PhysSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x, gamma_bias):
        x = x + self.attn(self.ln_1(x), gamma_bias=gamma_bias)
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x, gamma_bias=None):
        if self.use_checkpoint and self.training:
            return checkpoint(
                lambda inp: self._forward_impl(inp, gamma_bias),
                x,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(x, gamma_bias)


class CrossAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        ctx_dim: int,
        out_dim: int,
        n_head: int,
        bias: bool,
        dropout: float,
    ):
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
        B, Nq, _ = q.shape
        _, P, _ = C.shape
        Q = self.q_proj(q).view(B, Nq, self.n_head, self.dk).transpose(1, 2)
        K = self.k_proj(C).view(B, P, self.n_head, self.dk).transpose(1, 2)
        V = self.v_proj(C).view(B, P, self.n_head, self.dk).transpose(1, 2)

        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        Y = att @ V
        Y = (
            Y.transpose(1, 2)
            .contiguous()
            .view(B, Nq, self.n_head * self.dk)
        )
        Y = self.o_proj(Y)
        return self.ln_out(Y)


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class SIRENLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega0=30.0, is_first=False, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.omega0 = omega0
        self.is_first = is_first
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_dim, 1 / in_dim)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / in_dim) / omega0,
                    math.sqrt(6 / in_dim) / omega0,
                )

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class FiLMHyper(nn.Module):
    def __init__(self, in_dim, layer_hidden: int, n_layers: int, width: int):
        super().__init__()
        self.n_layers = n_layers
        self.width = width
        out_dim = n_layers * (2 * width + width)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, layer_hidden),
            nn.GELU(),
            nn.Linear(layer_hidden, layer_hidden),
            nn.GELU(),
            nn.Linear(layer_hidden, out_dim),
        )

    def forward(self, g):
        B, Nq, _ = g.shape
        vec = self.mlp(g)
        chunks = torch.chunk(vec, self.n_layers, dim=-1)
        gammas, betas, omegas = [], [], []
        for ch in chunks:
            gw, bw, ow = torch.split(
                ch, [self.width, self.width, self.width], dim=-1
            )
            gammas.append(gw)
            betas.append(bw)
            omegas.append(F.softplus(ow) + 1e-2)
        return gammas, betas, omegas


class FiLMSIREN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        width: int,
        depth: int,
        omega0: float,
        hyper_in_dim: int,
        hyper_hidden: int,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.omega0 = omega0
        self.hyper = FiLMHyper(hyper_in_dim, hyper_hidden, depth, width)

        layers = []
        layers.append(SIRENLayer(in_dim, width, omega0=omega0, is_first=True))
        for _ in range(depth - 2):
            layers.append(
                SIRENLayer(width, width, omega0=omega0, is_first=False)
            )
        self.layers = nn.ModuleList(layers)

        self.final = nn.Linear(width, 1)
        with torch.no_grad():
            self.final.weight.uniform_(
                -math.sqrt(6 / width) / omega0, math.sqrt(6 / width) / omega0
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


class PATModel(nn.Module):
    def __init__(self, cfg: PATConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = nn.Linear(cfg.d_patch, cfg.n_embd, bias=cfg.bias)
        self.pos_enc = nn.Linear(cfg.d_pos, cfg.n_embd, bias=cfg.bias)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.n_embd) * 0.02)

        self.blocks = nn.ModuleList(
            [
                PATBlock(
                    cfg.n_embd,
                    cfg.n_head,
                    cfg.dropout,
                    cfg.bias,
                    cfg.use_gradient_checkpointing,
                )
                for _ in range(cfg.n_layer)
            ]
        )

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

        inr_width = 128
        inr_depth = 4
        self.inr = FiLMSIREN(
            in_dim=query_dim,
            width=inr_width,
            depth=inr_depth,
            omega0=30.0,
            hyper_in_dim=cfg.n_embd + cfg.n_embd,
            hyper_hidden=256,
        )

        self.register_buffer("neg_inf", torch.tensor(float("-inf")))

    def heat_kernel_bias(
        self, ctx_pos: torch.Tensor
    ) -> torch.Tensor:
        B, P, _ = ctx_pos.shape
        x = ctx_pos[..., 0].unsqueeze(-1)
        t = ctx_pos[..., 1].unsqueeze(-1)

        dx2 = (x - x.transpose(1, 2)) ** 2
        dt = t - t.transpose(1, 2)
        mask = dt > 0

        safe_dt = torch.clamp(dt, min=1e-6)
        safe_dx2 = torch.clamp(dx2, min=1e-10)

        log_term = -0.5 * torch.log(
            torch.clamp(4 * math.pi * self.cfg.nu_bar * safe_dt, min=1e-10)
        )
        exp_term = -safe_dx2 / torch.clamp(
            4 * self.cfg.nu_bar * safe_dt, min=1e-10
        )
        exp_term = torch.clamp(exp_term, min=-50, max=50)

        logG = log_term + exp_term
        logG = torch.clamp(logG, min=-50, max=50)

        logG = torch.where(mask, self.cfg.alpha * logG, self.neg_inf)
        Gamma = logG.unsqueeze(1)
        return Gamma

    def encode_context(
        self,
        ctx_feats: torch.Tensor,
        ctx_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = ctx_feats.shape

        e = self.patch_embed(ctx_feats) + self.pos_enc(ctx_pos)
        cls = self.cls.expand(B, -1, -1)
        C = torch.cat([cls, e], dim=1)

        gamma = self.heat_kernel_bias(ctx_pos)
        gamma_full = gamma.new_zeros(B, 1, P + 1, P + 1)
        gamma_full[:, :, 1:, 1:] = gamma

        for blk in self.blocks:
            C = blk(C, gamma_bias=gamma_full)

        C = self.ln_ctx(C)
        cglob = C[:, :1, :]
        C_ctx = C[:, 1:, :]
        return C_ctx, cglob

    def predict_u(self, C: torch.Tensor, cglob: torch.Tensor, xt_q: torch.Tensor):
        phi = self.ff(xt_q)
        g = self.cross(phi, C)
        u = self.inr(phi, g, cglob)
        return u

    def forward(
        self,
        ctx_feats: torch.Tensor,
        ctx_pos: torch.Tensor,
        xt_q: torch.Tensor,
    ):
        C, cglob = self.encode_context(ctx_feats, ctx_pos)
        u_q = self.predict_u(C, cglob, xt_q)
        return u_q

    def compute_pde_residual(self, u, xt, nu, f=None):
        if f is None:
            f = torch.zeros_like(u)

        if not xt.requires_grad:
            xt = xt.requires_grad_(True)

        grads = torch.autograd.grad(
            u,
            xt,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        dudx = grads[..., 0:1]
        dudt = grads[..., 1:2]

        d2udx2 = torch.autograd.grad(
            dudx,
            xt,
            grad_outputs=torch.ones_like(dudx),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0][..., 0:1]

        r = dudt - nu * d2udx2 - f
        return r

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type
    ):
        param_dict = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad
        }
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(
            torch.optim.AdamW
        ).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **({"fused": True} if use_fused else {}),
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer