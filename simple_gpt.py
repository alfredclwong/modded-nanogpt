# %%
"""TODO
- init weights
- rope+yarn
- flash attention
- swiglu/relu^2
- qk norm
- multi gpu
"""

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import huggingface_hub as hf
import numpy as np
import pandas as pd
import torch
import wandb
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
import einops


# %%
class CausalSelfAttention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, flash: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.w_qkv = torch.nn.Linear(dim, 3 * self.head_dim * num_heads, bias=False)
        self.w_o = torch.nn.Linear(self.head_dim * num_heads, dim, bias=False)

        self.dropout_p = dropout
        self.dropout = torch.nn.Dropout(dropout)

        self.flash = flash

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.w_qkv(x)
        qkv = qkv.view(B, T, self.num_heads, 3 * self.head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        dropout_p = self.dropout_p if self.training else 0.0
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            attn_scores = einops.einsum(
                q, k, "b t h d, b s h d -> b h t s"
            )
            attn_scores = attn_scores / (self.head_dim**0.5)
            causal_mask = torch.tril(
                torch.ones((T, T), device=x.device, dtype=torch.bool)
            )
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            y = einops.einsum(
                attn_weights, v, "b h t s, b s h d -> b t h d"
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.w_o(y)
        y = self.dropout(y)
        return y


class MLP(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, flash: bool):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, dropout, flash)
        self.mlp = MLP(dim, dim * 4, dropout)
        self.ln1 = torch.nn.LayerNorm(dim, bias=False)
        self.ln2 = torch.nn.LayerNorm(dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleGPT(torch.nn.Module, hf.PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        weight_tying: bool,
        dropout: float,
        flash: bool,
    ):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, dim)
        self.pos_emb = torch.nn.Parameter(torch.zeros(max_seq_len, dim))
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(dim, num_heads, dropout, flash) for _ in range(num_layers)]
        )
        self.ln_f = torch.nn.LayerNorm(dim, bias=False)
        self.head = torch.nn.Linear(dim, vocab_size, bias=False)
        if weight_tying:
            self.head.weight = self.token_emb.weight

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        B, T = x.size()
        token_embeddings = self.token_emb(x)
        x = token_embeddings + self.pos_emb[:T]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        x = self.head(x)
        output = {"logits": x}

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                x.view(-1, x.size(-1)),
                labels.view(-1),
            )
            output["loss"] = loss

        return output


def init_weights(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


# %%
class ShardedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        repo: str,
        files: Iterable[str],
        cache_dir: Path,
        seq_len: int,
        batch_size: int,
        device: str,
        n_batches: int | None = None,
        cleanup: bool = False,
        shuffle: bool = True,
    ):
        self.repo = repo
        self.files = files
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.cleanup = cleanup
        self.shuffle = shuffle

        self.shard_idx = 0
        self.data = None
        self.local_path = None
        self.load_shard()

        max_batches = len(self.data) // (self.seq_len * self.batch_size)
        self.n_batches = (
            len(self.data) // (self.seq_len * self.batch_size)
            if n_batches is None
            else min(n_batches, max_batches)
        )

    def load_shard(self):
        fname = self.files[self.shard_idx]
        self.local_path = hf_hub_download(
            repo_id=self.repo,
            filename=fname,
            repo_type="dataset",
            local_dir=self.cache_dir,
        )
        self.data = np.memmap(self.local_path, dtype=np.uint16, mode="r")

    def __iter__(self):
        while self.shard_idx < len(self.files):
            if self.data is None or self.local_path is None:
                self.load_shard()
            start_idxs = np.arange(0, len(self.data) - self.seq_len - 1, self.seq_len)
            if self.shuffle:
                np.random.shuffle(start_idxs)
            start_idxs = start_idxs[: self.n_batches * self.batch_size]
            for i in range(0, len(start_idxs), self.batch_size):
                batch_start_idxs = start_idxs[i : i + self.batch_size]
                x = torch.stack(
                    [
                        torch.from_numpy(
                            self.data[idx : idx + self.seq_len].astype(np.int64)
                        )
                        for idx in batch_start_idxs
                    ]
                )
                y = torch.stack(
                    [
                        torch.from_numpy(
                            self.data[idx + 1 : idx + 1 + self.seq_len].astype(np.int64)
                        )
                        for idx in batch_start_idxs
                    ]
                )
                if self.device == "cpu":
                    x = x.pin_memory().to(self.device, non_blocking=True)
                    y = y.pin_memory().to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device)
                    y = y.to(self.device)
                yield x, y
            if self.cleanup:
                Path(self.local_path).unlink(missing_ok=True)
            self.data = None
            self.local_path = None
            self.shard_idx += 1

    def __len__(self):
        return len(self.files) * self.n_batches


# %%
@dataclass(frozen=True)
class DataConfig:
    train_repo: str = "kjj0/fineweb10B-gpt2"
    train_files: tuple[str, ...] = tuple(
        f"fineweb_train_{i:06d}.bin" for i in range(1, 2)
    )
    # train_files: tuple[str] = tuple(f"fineweb_train_{i:06d}.bin" for i in range(102, 104))
    val_repo: str = "kjj0/fineweb10B-gpt2"
    val_files: tuple[str, ...] = ("fineweb_val_000000.bin",)
    cache_dir: Path = Path("./data/cache")
    seq_len: int = 1024
    batch_size: int = 12
    n_train_batches: int | None = None
    val_seq_len: int = 1024
    val_batch_size: int = 16
    n_val_batches: int | None = 64
    cleanup: bool = False
    shuffle: bool = True


@dataclass(frozen=True)
class TrainConfig:
    n_epochs: int = 1
    learning_rate: float = 3e-4
    gradient_accumulation_steps: int = 4
    val_steps: int | float = 0.05
    ckpt_steps: int | float = 0.1
    use_wandb: bool = False

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_seq_len: int = 1024
    weight_tying: bool = False
    dropout: float = 0.0
    flash: bool = False


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()


cfg = Config()


# %%
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
device

# %%
model = SimpleGPT(**cfg.model.__dict__)
model.apply(init_weights)
model.to(device)


# %%
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_max_memory_usage(model: torch.nn.Module, batch_size: int) -> float:
    num_params = count_parameters(model)
    bytes_per_param = 4  # assuming float32
    multiplier = 4  # params, gradients, optimizer states (momentum, variance)
    mb = batch_size * num_params * bytes_per_param * multiplier / (1024**2)
    return mb


num_params_m = count_parameters(model) // 1_000_000
print(f"{num_params_m}M params")

estimated_mem_mb = estimate_max_memory_usage(
    model,
    # batch_size=12,
    batch_size=cfg.data.batch_size,
)
print(f"Estimated max memory usage: {estimated_mem_mb:.2f} MB")


# %%
def evaluate_model(
    model: torch.nn.Module,
    val_dataset: ShardedDataset,
    shifted: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for val_batch in tqdm(val_dataset, desc="Evaluating..."):
            val_inputs, val_targets = val_batch
            val_outputs = model(
                val_inputs, labels=val_targets if shifted else val_inputs
            )
            total_loss += val_outputs["loss"].item()
    avg_loss = total_loss / len(val_dataset)
    model.train()
    return avg_loss


# %%
train_dataset = ShardedDataset(
    repo=cfg.data.train_repo,
    files=cfg.data.train_files,
    cache_dir=cfg.data.cache_dir,
    seq_len=cfg.data.seq_len,
    batch_size=cfg.data.batch_size,
    device=device,
    n_batches=cfg.data.n_train_batches,
    shuffle=cfg.data.shuffle,
)

# %%
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = DATA_DIR / "model"
SIMPLE_GPT_DIR = MODEL_DIR / "simple-gpt"
SIMPLE_GPT_DIR.mkdir(parents=True, exist_ok=True)

# %%
print(
    f"Effective batch size: {cfg.data.batch_size * cfg.train.gradient_accumulation_steps}"
)
step = 0
n_tokens_seen = 0
n_total_steps = cfg.train.n_epochs * len(train_dataset)
val_steps = (
    cfg.train.val_steps
    if isinstance(cfg.train.val_steps, int)
    else int(n_total_steps * cfg.train.val_steps)
)
ckpt_steps = (
    cfg.train.ckpt_steps
    if isinstance(cfg.train.ckpt_steps, int)
    else int(n_total_steps * cfg.train.ckpt_steps)
)
val_loss = float("nan")
optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

if cfg.train.use_wandb:
    wandb.init(
        project="nano-gpt",
        config=cfg.__dict__,
    )

for epoch in range(cfg.train.n_epochs):
    model.train()
    train_dataset = ShardedDataset(
        repo=cfg.data.train_repo,
        files=cfg.data.train_files,
        cache_dir=cfg.data.cache_dir,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        device=device,
        n_batches=cfg.data.n_train_batches,
        shuffle=cfg.data.shuffle,
    )
    next_val_step = val_steps
    next_ckpt_step = ckpt_steps
    pbar = tqdm(train_dataset)
    for batch in pbar:
        step += 1

        inputs, targets = batch
        n_tokens_seen += inputs.numel()
        outputs = model(inputs, labels=targets)

        loss = outputs["loss"]
        loss = loss / cfg.train.gradient_accumulation_steps
        loss.backward()
        if step % cfg.train.gradient_accumulation_steps == 0:
            optimiser.step()
            optimiser.zero_grad()

        if step >= next_val_step:
            next_val_step += val_steps
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_dataset = ShardedDataset(
                    repo=cfg.data.val_repo,
                    files=cfg.data.val_files,
                    cache_dir=cfg.data.cache_dir,
                    seq_len=cfg.data.val_seq_len,
                    batch_size=cfg.data.val_batch_size,
                    device=device,
                    n_batches=cfg.data.n_val_batches,
                )
                for val_batch in tqdm(val_dataset, desc="Validating...", leave=False):
                    val_inputs, val_targets = val_batch
                    val_outputs = model(val_inputs, labels=val_targets)
                    val_loss += val_outputs["loss"].item()
                val_loss /= len(val_dataset)
            if cfg.train.use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            model.train()

        unscaled_loss = loss.item() * cfg.train.gradient_accumulation_steps
        pbar.set_description(
            f"[Epoch {epoch + 1}/{cfg.train.n_epochs}] Loss: {unscaled_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        if cfg.train.use_wandb:
            wandb.log(
                {
                    "train/loss": unscaled_loss,
                    "lr": optimiser.param_groups[0]["lr"],
                },
                step=step,
            )

        if step >= next_ckpt_step:
            next_ckpt_step += ckpt_steps
            n_tokens_seen_m = n_tokens_seen // 1_000_000
            ckpt_dir = (
                SIMPLE_GPT_DIR / f"simple-gpt-{num_params_m}M-{n_tokens_seen_m}Mtoks"
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_directory=ckpt_dir)
model.save_pretrained(SIMPLE_GPT_DIR, push_to_hub=True, repo_id="awonga/simple-gpt")
if cfg.train.use_wandb:
    wandb.finish()


# %%
def model_summary(model: torch.nn.Module) -> pd.DataFrame:
    rows = []
    for name, param in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": list(param.shape),
                "num_params": param.numel(),
                "requires_grad": param.requires_grad,
                "dtype": str(param.dtype),
                "mb": param.numel() * param.element_size() / (1024**2),
            }
        )
    return pd.DataFrame(rows)


summary_df = model_summary(model)
print(f"Total parameters: {summary_df['num_params'].sum() / 1_000_000:.2f} M")
print(f"Total memory: {summary_df['mb'].sum():.2f} MB")
summary_df

# %%
