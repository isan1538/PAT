import math
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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
        self.first = SIRENLayer(in_dim, width, omega0=omega0, is_first=True)
        self.hidden = nn.ModuleList(
            [nn.Linear(width, width, bias=True) for _ in range(depth - 1)]
        )
        self.sine = Sine()
        self.out = nn.Linear(width, 1, bias=True)
        self.hyper = FiLMHyper(
            hyper_in_dim, hyper_hidden, n_layers=depth - 1, width=width
        )

    def forward(self, phi, g, cglob):
        B, Nq, _ = phi.shape
        if cglob is not None:
            cgb = cglob.expand(-1, Nq, -1)
            hyper_in = torch.cat([g, cgb], dim=-1)
        else:
            hyper_in = g

        gammas, betas, omegas = self.hyper(hyper_in)
        h = self.first(phi)

        for i, lin in enumerate(self.hidden):
            z = lin(h)
            z = gammas[i] * z + betas[i]
            z = omegas[i] * z
            h = self.sine(z)

        u = self.out(h)
        return u


class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, m=64, logspace=True):
        super().__init__()
        self.in_dim = in_dim
        self.m = m
        if logspace:
            exps = torch.linspace(0, 1, steps=m)
            freqs = (2.0 ** (10.0 * exps)).unsqueeze(0)
        else:
            freqs = torch.linspace(1.0, 2**10, steps=m).unsqueeze(0)
        self.register_buffer("B", torch.cat([freqs, freqs], dim=0))

    def forward(self, xt):
        B, Nq, _ = xt.shape
        proj = torch.matmul(xt, self.B)
        s = torch.sin(2 * math.pi * proj)
        c = torch.cos(2 * math.pi * proj)
        return torch.cat([xt, s, c], dim=-1)


@dataclass
class PATConfig:
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

    d_patch: int = 32

    d_query: int = 2 + 128 + 128
    d_cross: int = 256

    siren_width: int = 256
    siren_depth: int = 5
    siren_omega0: float = 30.0
    hyper_hidden: int = 256

    m_ff: int = 128

    use_uncertainty: bool = True
    lambda_spec: float = 0.0
    lambda_reg: float = 0.0

    x_min: float = 0.0
    x_max: float = 1.0

    use_gradient_checkpointing: bool = True


class PATModel(nn.Module):
    def __init__(self, cfg: PATConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = nn.Linear(cfg.d_patch, cfg.n_embd, bias=cfg.bias)
        self.pos_enc = nn.Linear(2, cfg.n_embd, bias=False)

        self.blocks = nn.ModuleList(
            [
                PATBlock(
                    cfg.n_embd,
                    cfg.n_head,
                    cfg.dropout,
                    cfg.bias,
                    use_checkpoint=cfg.use_gradient_checkpointing,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_ctx = LayerNorm(cfg.n_embd, bias=cfg.bias)

        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        self.cross = CrossAttention(
            q_dim=(2 + 2 * cfg.m_ff),
            ctx_dim=cfg.n_embd,
            out_dim=cfg.d_cross,
            n_head=cfg.n_head,
            bias=cfg.bias,
            dropout=cfg.dropout,
        )

        self.ff = FourierFeatures(in_dim=2, m=cfg.m_ff, logspace=True)

        hyper_in_dim = cfg.d_cross + cfg.n_embd
        self.inr = FiLMSIREN(
            in_dim=(2 + 2 * cfg.m_ff),
            width=cfg.siren_width,
            depth=cfg.siren_depth,
            omega0=cfg.siren_omega0,
            hyper_in_dim=hyper_in_dim,
            hyper_hidden=cfg.hyper_hidden,
        )

        if cfg.use_uncertainty:
            self.logsig_data = nn.Parameter(torch.zeros(1))
            self.logsig_pde = nn.Parameter(torch.zeros(1))
            self.logsig_bc = nn.Parameter(torch.zeros(1))
            self.logsig_ic = nn.Parameter(torch.zeros(1))

        self.register_buffer("neg_inf", torch.tensor(float("-inf")))

    def heat_kernel_bias(
        self, ctx_pos: torch.Tensor, alpha: float, nu_bar: float
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
            torch.clamp(4 * math.pi * nu_bar * safe_dt, min=1e-10)
        )
        exp_term = -safe_dx2 / torch.clamp(
            4 * nu_bar * safe_dt, min=1e-10
        )
        exp_term = torch.clamp(exp_term, min=-50, max=50)

        logG = log_term + exp_term
        logG = torch.clamp(logG, min=-50, max=50)

        logG = torch.where(mask, alpha * logG, self.neg_inf)
        Gamma = logG.unsqueeze(1)
        return Gamma

    def encode_context(
        self,
        ctx_feats: torch.Tensor,
        ctx_pos: torch.Tensor,
        alpha: float = 1.0,
        nu_bar: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = ctx_feats.shape

        e = self.patch_embed(ctx_feats) + self.pos_enc(ctx_pos)
        cls = self.cls.expand(B, -1, -1)
        C = torch.cat([cls, e], dim=1)

        gamma = self.heat_kernel_bias(ctx_pos, alpha, nu_bar)
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

    def _diffusion_residual(self, u, xt, f, nu):
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

    def _uncert_weight(self, loss, logsig):
        logsig_clamped = torch.clamp(logsig, -2, 2)

        weighted_loss = 0.5 * torch.exp(-2 * logsig_clamped) * loss
        regularizer = logsig_clamped

        total = weighted_loss + regularizer

        if torch.any(total.detach() < 0):
            return loss

        return total

    def forward(
        self,
        ctx_feats: torch.Tensor,
        ctx_pos: torch.Tensor,
        xt_q: torch.Tensor,
        f_q: Optional[torch.Tensor] = None,
        training_dict: Optional[Dict] = None,
        weight=1.0,
    ):
        C, cglob = self.encode_context(ctx_feats, ctx_pos)

        if training_dict is not None:
            xt_q = xt_q.requires_grad_(True)
        u_q = self.predict_u(C, cglob, xt_q)

        if training_dict is None:
            return {"u": u_q}

        B = xt_q.size(0)
        device = xt_q.device
        loss_data = torch.tensor(0.0, device=device)
        loss_pde = torch.tensor(0.0, device=device)
        loss_bc = torch.tensor(0.0, device=device)
        loss_ic = torch.tensor(0.0, device=device)

        nu = training_dict.get("nu", 0.1)

        if "hr" in training_dict and training_dict["hr"] is not None:
            xh = training_dict["hr"]["x"]
            yh = training_dict["hr"]["y"]
            uh = self.predict_u(C, cglob, xh)
            loss_hr = F.mse_loss(uh, yh)
        else:
            loss_hr = torch.tensor(0.0, device=device)

        if "lr" in training_dict and training_dict["lr"] is not None:
            ylr = training_dict["lr"]["y_lr"]
            if "quad_points" in training_dict["lr"]:
                quad_pts = training_dict["lr"]["quad_points"]
                B_lr, Nl, Nqp, _ = quad_pts.shape
                quad_flat = quad_pts.view(B_lr, Nl * Nqp, 2)
                u_quad = self.predict_u(C, cglob, quad_flat)
                u_quad = u_quad.view(B_lr, Nl, Nqp, 1)
                ulr = u_quad.mean(dim=2)
                loss_lr = F.mse_loss(ulr, ylr)
            else:
                loss_lr = torch.tensor(0.0, device=device)
        else:
            loss_lr = torch.tensor(0.0, device=device)
        loss_data = loss_hr + loss_lr

        if "pde" in training_dict and training_dict["pde"] is not None:
            xr = training_dict["pde"]["colloc"]
            fr = training_dict["pde"].get("f_r", None)
            if not xr.requires_grad:
                xr = xr.clone().requires_grad_(True)
            ur = self.predict_u(C, cglob, xr)
            rr = self._diffusion_residual(ur, xr, fr, nu)
            loss_pde = torch.log(1 + (rr**2).mean())

        if "bc" in training_dict and training_dict["bc"] is not None:
            bc = training_dict["bc"]
            x_min = self.cfg.x_min
            x_max = self.cfg.x_max

            if "a_times" in bc and "g_a" in bc:
                ta = bc["a_times"]
                xa = torch.cat(
                    [torch.full_like(ta, x_min), ta], dim=-1
                )
                ua = self.predict_u(C, cglob, xa)
                loss_bc_a = F.mse_loss(ua, bc["g_a"])
            else:
                loss_bc_a = torch.tensor(0.0, device=device)

            if "b_times" in bc and "g_b" in bc:
                tb = bc["b_times"]
                xb = torch.cat(
                    [torch.full_like(tb, x_max), tb], dim=-1
                )
                ub = self.predict_u(C, cglob, xb)
                loss_bc_b = F.mse_loss(ub, bc["g_b"])
            else:
                loss_bc_b = torch.tensor(0.0, device=device)

            loss_bc = loss_bc_a + loss_bc_b

        if "ic" in training_dict and training_dict["ic"] is not None:
            x0 = training_dict["ic"]["x0"]
            u0 = training_dict["ic"]["u0"]
            xt0 = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            u_hat0 = self.predict_u(C, cglob, xt0)
            loss_ic = F.mse_loss(u_hat0, u0)

        if self.cfg.use_uncertainty:
            loss = self._uncert_weight(loss_data, self.logsig_data) + (
                weight * self._uncert_weight(loss_pde, self.logsig_pde)
                + weight * self._uncert_weight(loss_bc, self.logsig_bc)
                + weight * self._uncert_weight(loss_ic, self.logsig_ic)
            )
        else:
            loss = loss_data + loss_pde + loss_bc + loss_ic

        return {
            "u": u_q,
            "loss": loss,
            "loss_components": {
                "data": loss_data.detach(),
                "pde": loss_pde.detach(),
                "bc": loss_bc.detach(),
                "ic": loss_ic.detach(),
            },
        }

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
