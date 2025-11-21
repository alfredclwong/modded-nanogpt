from dataclasses import dataclass
from functools import partial

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


class GPT(torch.nn.Module):
    def __init__(self, model_cfg: GPTConfig):
        super().__init__()
        self.token_emb = torch.nn.Embedding(model_cfg.vocab_size, model_cfg.dim)
        with torch.no_grad():
            torch.nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if not model_cfg.rope:
            self.pos_emb = torch.nn.Embedding(model_cfg.max_seq_len, model_cfg.dim)
            with torch.no_grad():
                torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        self.blocks = torch.nn.ModuleList(
            Block(
                model_cfg.dim,
                model_cfg.num_heads,
                model_cfg.norm,
                model_cfg.rope,
                model_cfg.qk_norm,
                model_cfg.act,
                torch.bfloat16 if model_cfg.bf16 else torch.float32,
            )
            for _ in range(model_cfg.num_layers)
        )
        self.ln_f = model_cfg.norm(model_cfg.dim, bias=False)
        self.head = torch.nn.Linear(model_cfg.dim, model_cfg.vocab_size, bias=False)
        with torch.no_grad():
            torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

        if model_cfg.bf16:
            for m in self.modules():
                if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):  # Parameters?
                    m.bfloat16()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)

        x = self.token_emb(x)
        if hasattr(self, "pos_emb"):
            x = x + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
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
        dim: int,
        num_heads: int,
        norm: type[torch.nn.Module] | partial,
        rope: bool,
        qk_norm: bool,
        act: type[torch.nn.Module],
        dtype: torch.dtype,
    ):
        super().__init__()
        self.norm1 = norm(dim, bias=False)
        self.attn = Attention(dim, num_heads, rope, qk_norm, norm, dtype)
        self.norm2 = norm(dim, bias=False)
        self.mlp = MLP(dim, act)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: bool,
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
        self.w_qkvo.label = "attn"

        with torch.no_grad():
            torch.nn.init.normal_(self.w_qkvo, mean=0.0, std=0.02)

        self.rope = rope
        if rope:
            self.rotary = Rotary(self.head_dim, dtype=dtype)

    def forward(self, x: torch.Tensor):
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
        if self.rope:
            cos, sin = self.rotary(T, device=x.device)
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
        self.c_fc.lr_mul = 2.  # to account for transpose
        self.act = act()
        self.c_proj = torch.nn.Parameter(torch.empty(dim, 4 * dim))  # match attn weights
        self.c_proj.label = "mlp"

        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.c_fc, nonlinearity="relu")
            torch.nn.init.kaiming_normal_(self.c_proj, nonlinearity="relu")

    def forward(self, x: torch.Tensor):
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
