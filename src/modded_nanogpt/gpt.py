from dataclasses import dataclass

import torch

from modded_nanogpt.util import next_multiple


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = next_multiple(50_257, 128)  # 50_304
    num_layers: int = 12
    num_heads: int = 6
    dim: int = 768
    max_seq_len: int = 2048


class GPT(torch.nn.Module):
    def __init__(self, model_cfg: GPTConfig):
        super().__init__()
        self.token_emb = torch.nn.Embedding(model_cfg.vocab_size, model_cfg.dim)
        self.pos_emb = torch.nn.Embedding(model_cfg.max_seq_len, model_cfg.dim)
        self.blocks = torch.nn.ModuleList(
            Block(model_cfg.dim, model_cfg.num_heads)
            for _ in range(model_cfg.num_layers)
        )
        self.ln_f = torch.nn.LayerNorm(model_cfg.dim, bias=False)
        self.head = torch.nn.Linear(model_cfg.dim, model_cfg.vocab_size, bias=False)

        torch.nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)

        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if y is not None:
            logits = self.head(x)
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
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dim, bias=False)
        self.attn = Attention(dim, num_heads)
        self.ln2 = torch.nn.LayerNorm(dim, bias=False)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.w_qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.w_o = torch.nn.Linear(dim, dim, bias=False)

        torch.nn.init.normal_(self.w_qkv.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w_o.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.size()
        qkv = self.w_qkv(x)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, T, head_dim)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # (B, num_heads, T, head_dim)
        attn_output = attn_output.permute(
            0, 2, 1, 3
        ).contiguous()  # (B, T, num_heads, head_dim)
        attn_output = attn_output.view(B, T, D)  # (B, T, D)
        output = self.w_o(attn_output)  # (B, T, D)
        return output


class MLP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = torch.nn.Linear(dim, 4 * dim, bias=False)
        self.c_proj = torch.nn.Linear(4 * dim, dim, bias=False)

        torch.nn.init.kaiming_normal_(self.c_fc.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.c_proj.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = torch.nn.functional.relu(x)
        x = self.c_proj(x)
        return x
