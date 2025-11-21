import uuid
from dataclasses import dataclass

import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

from modded_nanogpt.data import distributed_data_generator
from modded_nanogpt.eval import Clock, eval
from modded_nanogpt.gpt import GPT
from modded_nanogpt.opt import DistAdam
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
    vals_per_ckpt: int
    use_wandb: bool


def train(model: GPT, train_cfg: TrainConfig, device: str):
    model.train()

    train_loader = distributed_data_generator(
        filename_pattern=train_cfg.train_files,
        batch_tokens=train_cfg.train_batch_tokens,
        max_seq_len=train_cfg.train_max_seq_len,
        grad_accum_steps=train_cfg.grad_accum_steps,
        device=device,
    )
    opt_kwargs = dict(
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )
    if dist.is_initialized():
        optimiser = DistAdam(model.parameters(), **opt_kwargs)
    else:
        optimiser = torch.optim.AdamW(model.parameters(), **opt_kwargs)

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
            val_loss = eval(
                model,
                filename_pattern=train_cfg.val_files,
                val_tokens=train_cfg.val_tokens,
                batch_tokens=train_cfg.val_batch_tokens,
                max_seq_len=train_cfg.train_max_seq_len,
                grad_accum_steps=train_cfg.grad_accum_steps,
                device=device,
            )
            if dist.is_initialized():
                dist.barrier()
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss = val_loss.item()
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
            if (
                train_cfg.vals_per_ckpt > 0
                and step % (train_cfg.vals_per_ckpt * train_cfg.val_steps) == 0
            ):
                torch.save(model.state_dict(), f"checkpoint_{step}.pt")
            if last_step:
                break
            clock.start()

        # --------------- TRAINING SECTION -----------------
        batch_loss = torch.tensor(0.0, device=device)
        for i in tqdm(
            range(train_cfg.grad_accum_steps),
            desc="Gradient Accumulation",
            total=train_cfg.grad_accum_steps,
            leave=False,
        ):
            if isinstance(optimiser, DistAdam) and i == train_cfg.grad_accum_steps - 1:
                optimiser.should_sync = True
            inputs, targets = next(train_loader)
            _, loss = model(inputs, targets)
            loss = loss / train_cfg.grad_accum_steps
            batch_loss += loss.detach()
            loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        if isinstance(optimiser, DistAdam):
            optimiser.should_sync = False
        if train_cfg.use_wandb:
            wandb.log({"train/loss": batch_loss.item()}, step=step)
    if train_cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import os
    import subprocess
    import sys
    import traceback
    from functools import partial

    from modded_nanogpt.gpt import GPTConfig, ReLU2, RMSNorm
    from modded_nanogpt.util import is_cuda

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and is_cuda(device):
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    run_id = str(uuid.uuid4())[:8]
    logfile = None
    if master_process:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, f"{run_id}.txt")
        print(f"Logging to {logfile}")

    def print0(s, console=True):
        s = str(s)
        if master_process:
            if console:
                print(s)
            with open(logfile, "a") as f:
                f.write(s + "\n")

    def nvidia_smi():
        return subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout

    try:
        print0(f"Python {sys.version}")
        print0(f"PyTorch {torch.__version__}")
        if is_cuda(device):
            print0(f"CUDA {torch.version.cuda}")
            print0(nvidia_smi())
        print0("=" * 80)

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

        print0("Compiling model...")
        model = torch.compile(model, dynamic=True, fullgraph=True)
        print0("Model compiled.")

        if dist.is_initialized():
            for param in model.parameters():
                dist.broadcast(param.detach(), src=0)

        # reduces train/val steps, 1 = full training
        DEBUG_FACTOR = 8

        # reduces mini batch size and sequence length (if > 16),
        # increases grad accum steps to keep tokens per batch constant
        VRAM_FACTOR = 2

        GRAD_ACCUM_STEPS = 8 * VRAM_FACTOR
        MINI_BATCH_SIZE = max(1, 16 // VRAM_FACTOR)
        MAX_SEQ_LEN = 2048 if MINI_BATCH_SIZE > 1 else 2048 * 16 // VRAM_FACTOR

        TRAIN_STEPS = 2245

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
            num_steps=TRAIN_STEPS // DEBUG_FACTOR,
            lr=3e-4,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            # eval and logging
            val_steps=TRAIN_STEPS // 20 // DEBUG_FACTOR,  # 0 for only at end
            vals_per_ckpt=5,  # 0 for only at end
            use_wandb=False and master_process,
        )

        print0(f"{device=}")
        print0(
            f"VRAM_FACTOR={VRAM_FACTOR} GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} MINI_BATCH_SIZE={MINI_BATCH_SIZE}"
            f" MAX_SEQ_LEN={MAX_SEQ_LEN} TRAIN_BATCH_TOKENS={TRAIN_BATCH_TOKENS} TRAIN_TOKENS={TRAIN_BATCH_TOKENS * train_cfg.num_steps}"
        )
        print0(model_cfg.__dict__ | train_cfg.__dict__)
        print0(model)

        train(model, train_cfg, device)
        print0("Training complete.")

        max_memory_used = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if is_cuda(device)
            else 0.0
        )
        print0(f"Max memory used: {max_memory_used:.2f} GB")

    except Exception:
        if master_process:
            print0("An exception occurred during training:")
            print0(traceback.format_exc())
        sys.exit(1)

    finally:
        print(logfile)
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.save(model.state_dict(), "final.pt")
