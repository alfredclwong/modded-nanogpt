import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

import wandb
from modded_nanogpt.data import data_generator
from modded_nanogpt.gpt import GPT
from modded_nanogpt.util import next_multiple


@dataclass(frozen=True)
class TrainConfig:
    # data
    train_files: str
    val_files: str
    train_batch_tokens: int
    train_max_seq_len: int
    grad_accum_steps: int
    val_tokens: int
    val_batch_tokens: int

    # optimisation
    num_steps: int
    lr: float
    weight_decay: float
    betas: tuple

    # eval and logging
    val_steps: int
    save_checkpoint: bool
    use_wandb: bool


class Clock:
    def __init__(self, device: str):
        self.elapsed_ms = 0.0
        if device.startswith("cuda"):
            self.sync_fn = torch.cuda.synchronize
        elif device.startswith("mps"):
            self.sync_fn = torch.mps.synchronize
        else:
            self.sync_fn = lambda: None

    def start(self):
        self.sync_fn()
        self.t0 = time.perf_counter()

    def pause(self):
        self.sync_fn()
        t1 = time.perf_counter()
        self.elapsed_ms += (t1 - self.t0) * 1000.0


def train(model: GPT, train_cfg: TrainConfig, device: str):
    model.train()

    train_loader = data_generator(
        filename_pattern=train_cfg.train_files,
        batch_tokens=train_cfg.train_batch_tokens,
        max_seq_len=train_cfg.train_max_seq_len,
        grad_accum_steps=train_cfg.grad_accum_steps,
        device=device,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )

    if train_cfg.use_wandb:
        wandb.init(
            project="modded-nanogpt",
            config={
                **train_cfg.__dict__,
                **model_cfg.__dict__,
            },
        )

    clock = Clock(device)
    clock.start()

    for step in tqdm(
        range(train_cfg.num_steps + 1), desc="Training", total=train_cfg.num_steps + 1
    ):
        last_step = step == train_cfg.num_steps

        # --------------- VALIDATION SECTION -----------------
        if last_step or (train_cfg.val_steps > 0 and step % train_cfg.val_steps == 0):
            clock.pause()
            val_loss = eval(model, train_cfg, device)
            print(
                "\n"
                f"{step=}/{train_cfg.num_steps}"
                f" {val_loss=:.4f} {clock.elapsed_ms=:.0f}ms"
                f" step_avg={clock.elapsed_ms / max(1, step):.2f}ms"
            )
            if train_cfg.use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "elapsed_ms": clock.elapsed_ms,
                        "step_avg_ms": clock.elapsed_ms / max(1, step),
                    },
                    step=step,
                )
            if train_cfg.save_checkpoint:
                torch.save(model.state_dict(), f"checkpoint_{step}.pt")
            if last_step:
                break
            clock.start()

        # --------------- TRAINING SECTION -----------------
        optimiser.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        for _ in tqdm(
            range(train_cfg.grad_accum_steps),
            desc="Gradient Accumulation",
            total=train_cfg.grad_accum_steps,
            leave=False,
        ):
            inputs, targets = next(train_loader)
            _, loss = model(inputs, targets)
            loss = loss / train_cfg.grad_accum_steps
            batch_loss += loss.detach()
            loss.backward()
        optimiser.step()
        if train_cfg.use_wandb:
            wandb.log({"train/loss": batch_loss.item()}, step=step)
    if train_cfg.use_wandb:
        wandb.finish()


@torch.no_grad()
def eval(model: GPT, train_cfg: TrainConfig, device: str) -> float:
    model.eval()

    val_loss = 0.0
    val_loader = data_generator(
        filename_pattern=train_cfg.val_files,
        batch_tokens=train_cfg.val_batch_tokens,
        max_seq_len=train_cfg.train_max_seq_len,
        grad_accum_steps=train_cfg.grad_accum_steps,
        device=device,
    )

    # 1 val step = 1 mini batch (vs 1 train step = grad_accum_steps mini batches)
    val_steps = (
        train_cfg.grad_accum_steps * train_cfg.val_tokens // train_cfg.val_batch_tokens
    )
    for _ in tqdm(range(val_steps), desc="Validation", total=val_steps, leave=False):
        inputs, targets = next(val_loader)
        _, loss = model(inputs, targets)
        val_loss += loss.item()
    val_loss /= val_steps

    model.train()
    return val_loss


if __name__ == "__main__":
    from modded_nanogpt.gpt import GPTConfig, RMSNorm, ReLU2
    from functools import partial

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model_cfg = GPTConfig(
        vocab_size=next_multiple(50_257, 128),  # 50_304
        num_layers=12,
        num_heads=6,
        dim=768,
        max_seq_len=2048,
        norm=partial(RMSNorm, elementwise_affine=False),
        rope=True,
        qk_norm=True,
        act=ReLU2,
        bf16=True,
    )
    model = GPT(model_cfg).to(device)

    DEBUG_FACTOR = 8  # reduces train/val steps, 1 = full training
    VRAM_FACTOR = 32  # reduces mini batch size and sequence length (if > 16),
                      # increases grad accum steps to keep tokens per batch constant

    GRAD_ACCUM_STEPS = 8 * VRAM_FACTOR
    MINI_BATCH_SIZE = max(1, 16 // VRAM_FACTOR)
    MAX_SEQ_LEN = 2048 if MINI_BATCH_SIZE > 1 else 2048 * 16 // VRAM_FACTOR

    TRAIN_BATCH_TOKENS = MAX_SEQ_LEN * MINI_BATCH_SIZE * GRAD_ACCUM_STEPS
    train_cfg = TrainConfig(
        # data
        train_files="data/fineweb10B/fineweb_train_*.bin",
        val_files="data/fineweb10B/fineweb_val_*.bin",
        train_batch_tokens=TRAIN_BATCH_TOKENS,
        train_max_seq_len=MAX_SEQ_LEN,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        val_tokens=10_485_760 // DEBUG_FACTOR,
        val_batch_tokens=TRAIN_BATCH_TOKENS,
        # optimisation
        num_steps=2245 // DEBUG_FACTOR,
        lr=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        # eval and logging
        val_steps=250 // DEBUG_FACTOR,  # 0 for only at end
        save_checkpoint=True,
        use_wandb=False,
    )

    print(f"{device=}")
    print(
        f"VRAM_FACTOR={VRAM_FACTOR} GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} MINI_BATCH_SIZE={MINI_BATCH_SIZE}"
        f" MAX_SEQ_LEN={MAX_SEQ_LEN} TRAIN_BATCH_TOKENS={TRAIN_BATCH_TOKENS} TRAIN_TOKENS={TRAIN_BATCH_TOKENS * train_cfg.num_steps}"
    )
    print(model_cfg.__dict__ | train_cfg.__dict__)
    print(model)

    train(model, train_cfg, device)
    torch.save(model.state_dict(), "final.pt")

    print("Training complete.")

    max_memory_used = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.startswith("cuda")
        else 0.0
    )
    print(f"Max memory used: {max_memory_used:.2f} GB")

