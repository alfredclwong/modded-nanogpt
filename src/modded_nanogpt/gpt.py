from dataclasses import dataclass
from functools import partial

import einops
import torch


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    num_layers: int
    num_heads: int
    dim: int
    max_seq_len: int
    norm: type[torch.nn.Module] | partial
    rope: bool
    qk_norm: bool
    act: type[torch.nn.Module]
    bf16: bool
    hc: bool
    dynamic: bool
    expansion_rate: int  # positive for HC, negative for FC
    dnorm: type[torch.nn.Module] | partial
    # technically these are training cfgs
    shc_lr_mul: float
    dhc_lr_mul: float


class GPT(torch.nn.Module):
    def __init__(self, model_cfg: GPTConfig):
        super().__init__()
        dtype = torch.bfloat16 if model_cfg.bf16 else torch.float32
        self.token_emb = torch.nn.Embedding(model_cfg.vocab_size, model_cfg.dim)
        self.rope = model_cfg.rope
        if self.rope:
            head_dim = model_cfg.dim // model_cfg.num_heads
            self.rotary = Rotary(head_dim, dtype=dtype)
        else:
            self.pos_emb = torch.nn.Embedding(model_cfg.max_seq_len, model_cfg.dim)
        block_cls = (
            (DHCBlock if model_cfg.dynamic else SHCBlock) if model_cfg.hc else Block
        )
        self.hc = model_cfg.hc
        self.expansion_rate = model_cfg.expansion_rate if self.hc else 0
        self.n = abs(self.expansion_rate)
        self.blocks = torch.nn.ModuleList(
            block_cls(
                layer_idx,
                model_cfg.dim,
                model_cfg.num_heads,
                model_cfg.norm,
                model_cfg.qk_norm,
                model_cfg.act,
                dtype,
                expansion_rate=self.expansion_rate,
                shc_lr_mul=model_cfg.shc_lr_mul,
                dhc_lr_mul=model_cfg.dhc_lr_mul,
                dnorm=model_cfg.dnorm,
            )
            for layer_idx in range(model_cfg.num_layers * 2)
        )
        self.ln_f = model_cfg.norm(model_cfg.dim, bias=False)
        self.head = torch.nn.Linear(model_cfg.dim, model_cfg.vocab_size, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()

        if model_cfg.bf16:
            for m in self.modules():
                if isinstance(m, (torch.nn.Embedding, torch.nn.Linear, torch.nn.Parameter)):
                    m.bfloat16()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)

        x = self.token_emb(x)
        if hasattr(self, "pos_emb"):
            x = x + self.pos_emb(pos)
        if self.hc:
            if self.expansion_rate > 0:
                x = einops.repeat(x, "b t d -> b t n d", n=self.n)
            else:
                x = einops.rearrange(x, "b t (n f) -> b t n f", n=self.n)
        if self.rope:
            cos, sin = self.rotary(T, x.device)
        else:
            cos, sin = None, None
        for block in self.blocks:
            x = block(x, cos=cos, sin=sin)
        if self.hc:
            if self.expansion_rate > 0:
                x = x.sum(dim=-2)  # sum over hyper-dim
            else:
                x = x.flatten(-2)  # flatten frac-dim
        x = self.ln_f(x)

        if y is not None:
            logits = self.head(x).to(torch.float32)
            loss = torch.nn.functional.cross_entropy(
                logits.view(B * T, -1), y.view(B * T), reduction="mean"
            )
        else:
            logits = self.head(
                x[:, [-1], :]
            )  # inference: return logits for last token only
            loss = None
        return logits, loss


class Block(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        num_heads: int,
        norm: type[torch.nn.Module] | partial,
        qk_norm: bool,
        act: type[torch.nn.Module],
        dtype: torch.dtype,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm(dim, bias=False)
        if layer_idx % 2 == 0:
            self.fn = Attention(dim, num_heads, qk_norm, norm, dtype)
        else:
            self.fn = MLP(dim, act)

    def forward(self, x: torch.Tensor, **kwargs):
        return x + self.fn(self.norm(x), **kwargs)


class SHCBlock(Block):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        num_heads: int,
        norm: type[torch.nn.Module] | partial,
        qk_norm: bool,
        act: type[torch.nn.Module],
        dtype: torch.dtype,
        expansion_rate: int,
        shc_lr_mul: float,
        **kwargs,
    ):
        super().__init__(
            layer_idx,
            dim,
            num_heads,
            norm,
            qk_norm,
            act,
            dtype,
        )
        self.expansion_rate = expansion_rate
        self.n = abs(expansion_rate)
        # hc is transposed vs the original paper for optimiser chunking
        self.hc = torch.nn.Parameter(
            torch.empty(self.n + 1, self.n + 1)
            if self.expansion_rate > 0
            else torch.empty(self.n * 2, self.n + 1)
        )
        self.hc.label = "shc"  # type: ignore
        self.hc.lr_mul = shc_lr_mul  # type: ignore
        with torch.no_grad():
            # top left
            self.hc[: -self.n, 0] = 0.0
            # bot left
            self.hc[-self.n :, 0] = 1.0
            # top right
            if expansion_rate > 0:
                self.hc[0, 1:] = 0.0
                self.hc[0, layer_idx % self.n + 1] = 1.0
            else:
                self.hc[: self.n, -self.n :] = torch.eye(self.n)
            # bot right
            self.hc[-self.n :, -self.n :] = torch.eye(self.n)
        print(f"hc ({layer_idx=}):\n{self.hc}")

    def forward(self, x: torch.Tensor, **kwargs):
        # x shape (B, T, n, D)
        A = self.hc[:, -self.n :].type_as(x)  # (n + 1, n) or (2n, n)
        B = self.hc[-self.n :, 0].type_as(x)  # (n,)
        hH = torch.einsum("pn,btnd->btpd", A, x)  # width connection (p = n + 1 or 2n)
        h = hH[..., : self.n, :].flatten(-2)
        H = hH[..., -self.n :, :]
        h = self.fn(self.norm(h), **kwargs)
        H = H + torch.einsum("n,btd->btnd", B, h)  # depth connection
        return H


class DHCBlock(SHCBlock):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        num_heads: int,
        norm: type[torch.nn.Module] | partial,
        qk_norm: bool,
        act: type[torch.nn.Module],
        dtype: torch.dtype,
        expansion_rate: int,
        shc_lr_mul: float,
        dhc_lr_mul: float,
        dnorm: type[torch.nn.Module] | partial,
    ):
        super().__init__(
            layer_idx,
            dim,
            num_heads,
            norm,
            qk_norm,
            act,
            dtype,
            expansion_rate,
            shc_lr_mul,
        )
        self.frac_dim = dim // self.n if expansion_rate < 0 else dim
        self.dnorm = dnorm(self.frac_dim, bias=False)
        self.s_a = torch.nn.Parameter(torch.empty(1))
        self.s_b = torch.nn.Parameter(torch.empty(1))
        self.w_a = torch.nn.Parameter(
            torch.empty(self.frac_dim, self.n + 1 if self.expansion_rate > 0 else self.n * 2)
        )
        self.w_b = torch.nn.Parameter(torch.empty(self.frac_dim))
        self.s_a.label = "dhc"  # type: ignore
        self.s_b.label = "dhc"  # type: ignore
        self.w_a.label = "dhc"  # type: ignore
        self.w_b.label = "dhc"  # type: ignore
        self.s_a.lr_mul = dhc_lr_mul  # type: ignore
        self.s_b.lr_mul = dhc_lr_mul  # type: ignore
        self.w_a.lr_mul = dhc_lr_mul  # type: ignore
        self.w_b.lr_mul = dhc_lr_mul  # type: ignore
        with torch.no_grad():
            self.s_a.fill_(1e-2)
            self.s_b.fill_(1e-2)
            torch.nn.init.zeros_(self.w_a)
            torch.nn.init.zeros_(self.w_b)

    def forward(self, x: torch.Tensor, **kwargs):
        # x shape (B, T, n, D) or (B, T, n, F)
        A = self.hc[:, -self.n :]  # (n + 1, n) or (2n, n)
        B = self.hc[-self.n :, 0]  # (n,)
        H_norm = self.dnorm(x.float())
        A = A + self.s_a * torch.nn.functional.tanh(
            H_norm @ self.w_a
        ).transpose(-2, -1)  # (B, T, n + 1, n) or (B, T, 2n, 2)
        B = B + self.s_b * torch.nn.functional.tanh(H_norm @ self.w_b)  # (B, T, n)
        hH = torch.einsum("btpn,btnd->btpd", A, x.float())  # width connection
        if self.expansion_rate > 0:
            h = hH[..., 0, :]
        else:
            h = hH[..., : self.n, :].flatten(-2)
        H = hH[..., -self.n :, :]
        h = self.fn(self.norm(h), **kwargs)
        if self.expansion_rate > 0:
            H = H + torch.einsum("btn,btd->btnd", B, h)  # depth connection
        else:
            h = einops.rearrange(h, "b t (n f) -> b t n f", n=self.n)
            H = H + torch.einsum("btn,btnf->btnf", B, h)  # depth connection
        return H.type_as(x)


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool,
        norm: type[torch.nn.Module] | partial,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = norm(self.head_dim, bias=False) if qk_norm else None

        # collate qkvo to be same size as mlp weights for optimiser param grouping
        self.w_qkvo = torch.nn.Parameter(torch.empty(dim, 4 * dim))
        self.w_qkvo.label = "attn"  # type: ignore

        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.w_qkvo.view(4, dim, dim)[:3].uniform_(-bound, bound)
            self.w_qkvo.view(4, dim, dim)[3].zero_()

    def forward(self, x: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None):
        B, T, D = x.size()
        qkv = torch.nn.functional.linear(
            x, self.w_qkvo.view(4, D, D)[:3].flatten(0, 1).type_as(x)
        )  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, T, head_dim)
        if self.qk_norm is not None:
            q = self.qk_norm(q)
            k = self.qk_norm(k)
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # (B, num_heads, T, head_dim)
        attn_output = attn_output.permute(
            0, 2, 1, 3
        ).contiguous()  # (B, T, num_heads, head_dim)
        attn_output = attn_output.view(B, T, D)  # (B, T, D)
        output = torch.nn.functional.linear(
            attn_output, self.w_qkvo.view(4, D, D)[3].type_as(x)
        )  # (B, T, D)
        return output


# https://blog.eleuther.ai/rotary-embeddings/
class Rotary(torch.nn.Module):
    def __init__(self, dim, dtype, base=10000):
        super().__init__()
        self.dtype = dtype
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len: int, device: torch.device | str):
        if self.seq_len_cached is None or self.seq_len_cached < seq_len:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self.cos_cached = emb.cos().to(self.dtype)
            self.sin_cached = emb.sin().to(self.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q_BHTD, k_BHTD, cos_TH, sin_TH):
    cos = cos_TH[None, None, :, :]  # (1, 1, T, head_dim)
    sin = sin_TH[None, None, :, :]  # (1, 1, T, head_dim)
    return (
        (q_BHTD * cos) + (rotate_half(q_BHTD) * sin),
        (k_BHTD * cos) + (rotate_half(k_BHTD) * sin),
    )


class MLP(torch.nn.Module):
    def __init__(self, dim: int, act: type[torch.nn.Module]):
        super().__init__()
        self.c_fc = torch.nn.Parameter(torch.empty(dim, 4 * dim))
        self.c_fc.label = "mlp"
        # self.c_fc.lr_mul = 2.  # to account for transpose?
        self.act = act()
        self.c_proj = torch.nn.Parameter(
            torch.empty(dim, 4 * dim)
        )  # match attn weights
        self.c_proj.label = "mlp"

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: torch.Tensor, **kwargs):
        x = torch.nn.functional.linear(x, self.c_fc.T.type_as(x))
        x = self.act(x)
        x = torch.nn.functional.linear(x, self.c_proj.type_as(x))
        return x


class ReLU2(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.relu(x).square()


class RMSNorm(torch.nn.RMSNorm):
    # allow dummy bias argument for consistency
    def __init__(self, dim: int, *, elementwise_affine: bool, bias: bool):
        super().__init__(dim, elementwise_affine=elementwise_affine)


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, dim: int, *, elementwise_affine: bool, bias: bool):
        super().__init__(dim, elementwise_affine=elementwise_affine, bias=bias)
