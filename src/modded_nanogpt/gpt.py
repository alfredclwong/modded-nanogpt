from dataclasses import dataclass
from functools import partial

import einops
import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)
# torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

from modded_nanogpt.util import is_mps


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
    window_size: int
    kernel_options: dict | None
    # technically these are training cfgs
    shc_lr_mul: float
    dhc_lr_mul: float


class GPT(torch.nn.Module):
    def __init__(self, model_cfg: GPTConfig):
        super().__init__()
        dtype = torch.bfloat16 if model_cfg.bf16 else torch.float32
        self.token_emb = torch.nn.Embedding(model_cfg.vocab_size, model_cfg.dim)
        with torch.no_grad():
            self.token_emb.weight.normal_(0, 0.02)
        self.rope = model_cfg.rope
        if self.rope:
            head_dim = model_cfg.dim // model_cfg.num_heads
            self.rotary = Rotary(head_dim, dtype=dtype)
        else:
            self.pos_emb = torch.nn.Embedding(model_cfg.max_seq_len, model_cfg.dim)
            with torch.no_grad():
                self.pos_emb.weight.normal_(0, 0.02)
        self.max_seq_len = model_cfg.max_seq_len
        self.window_size = model_cfg.window_size
        self.kernel_options = model_cfg.kernel_options
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
            self.head.weight.normal_(0, 0.02)

        if model_cfg.bf16:
            for m in self.modules():
                if isinstance(
                    m, (torch.nn.Embedding, torch.nn.Linear, torch.nn.Parameter)
                ):
                    m.bfloat16()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, seqlens: torch.Tensor
    ) -> torch.Tensor:
        B, T = inputs.size()
        assert B == 1, f"Expect batch size of 1, got {inputs.shape=}"
        device = inputs.device

        x = self.token_emb(inputs)
        if hasattr(self, "pos_emb"):
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        if self.hc:
            if self.expansion_rate > 0:
                x = einops.repeat(x, "b t d -> b t n d", n=self.n)
            else:
                x = einops.rearrange(x, "b t (n f) -> b t n f", n=self.n)
        if self.rope:
            cos, sin = self.rotary(T, device)
        else:
            cos, sin = None, None

        doc_ids = (seqlens.unsqueeze(1) <= torch.arange(T, device=device)).sum(0)
        def sliding_window_causal_mask(b, h, q_idx, kv_idx, window_size):
            causal_mask = q_idx >= kv_idx
            # Tokens can only attend within same document
            same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
            if window_size is None:
                return causal_mask & same_doc
            # Apply sliding window: can only attend to tokens within window_size
            window_mask = (q_idx - kv_idx) <= window_size
            return causal_mask & same_doc & window_mask

        block_masks = [
            create_block_mask(
                partial(sliding_window_causal_mask, window_size=ws),
                B=B,
                H=None,
                Q_LEN=T,
                KV_LEN=T,
                device=device,
                _compile=True,
            )
            for ws in [None, self.window_size, self.window_size // 2]
        ]  # full, short, long
        for i, block in enumerate(self.blocks):
            if i in {0, 7}:
                block_mask = block_masks[0]  # full
            elif i in {4, 11}:
                block_mask = block_masks[1]  # long
            else:
                block_mask = block_masks[2]  # short
            x = block(
                x,
                cos=cos,
                sin=sin,
                block_mask=block_mask,
                kernel_options=self.kernel_options,
            )

        if self.hc:
            if self.expansion_rate > 0:
                x = x.sum(dim=-2)  # sum over hyper-dim
            else:
                x = x.flatten(-2)  # flatten frac-dim
        x = self.ln_f(x)

        logits = self.head(x)
        logits = logits.float() if self.training else logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum" if self.training else "mean",
        )
        return loss


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
            torch.empty(
                self.frac_dim, self.n + 1 if self.expansion_rate > 0 else self.n * 2
            )
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
            self.w_a.zero_()
            self.w_b.zero_()

    def forward(self, x: torch.Tensor, **kwargs):
        # x shape (B, T, n, D) or (B, T, n, F)
        A = self.hc[:, -self.n :]  # (n + 1, n) or (2n, n)
        B = self.hc[-self.n :, 0]  # (n,)
        H_norm = self.dnorm(x.float())
        A = A + self.s_a * torch.nn.functional.tanh(H_norm @ self.w_a).transpose(
            -2, -1
        )  # (B, T, n + 1, n) or (B, T, 2n, 2)
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

        with torch.no_grad():
            self.w_qkvo.normal_(0, 0.02)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        block_mask: BlockMask,
        kernel_options: dict | None,
    ) -> torch.Tensor:
        B, T, D = x.size()
        qkv = torch.nn.functional.linear(
            x, self.w_qkvo.view(4, D, D)[:3].flatten(0, 1).type_as(x)
        )  # (B, T, 3*D)
        q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )  # (B, num_heads, T, head_dim)
        if self.qk_norm is not None:
            q = self.qk_norm(q)
            k = self.qk_norm(k)
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if B > 1 or is_mps(x.device):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )  # (B, num_heads, T, head_dim)
        else:
            attn_output = flex_attention(
                q, k, v, block_mask=block_mask, kernel_options=kernel_options
            )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
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

        with torch.no_grad():
            self.c_fc.normal_(0, 0.02)
            self.c_proj.normal_(0, 0.02)

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
