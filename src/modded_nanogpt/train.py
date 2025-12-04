import uuid
from collections import Counter
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tqdm import tqdm

import wandb
from modded_nanogpt.data0 import distributed_data_generator
from modded_nanogpt.eval import Clock, eval
from modded_nanogpt.gpt import GPT
from modded_nanogpt.opt import DistAdam, OptimConfig
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
    adam_cfg: OptimConfig
    muon_cfg: OptimConfig

    # eval and logging
    val_steps: int
    vals_per_ckpt: int
    use_wandb: bool


def train(model: GPT, train_cfg: TrainConfig, device: str | torch.device):
    model.train()

    train_loader = distributed_data_generator(
        filename_pattern=train_cfg.train_files,
        num_tokens=train_cfg.train_batch_tokens,
        max_seq_len=train_cfg.train_max_seq_len,
        grad_accum_steps=train_cfg.grad_accum_steps,
        align_to_bos=True,
        device=device,
    )

    adam_params = []
    muon_params = []
    for n, p in model.named_parameters():
        if any(x == model_cfg.vocab_size for x in p.shape) or p.ndim < 2:
            adam_params.append(p)
        elif p.label in ["dhc", "shc"]:
            muon_params.append(p)
        else:
            muon_params.append(p)
    adam_param_shapes = Counter([tuple(p.shape) for p in adam_params])
    muon_param_shapes = Counter([tuple(p.shape) for p in muon_params])
    print(f"Adam param shapes: {adam_param_shapes}")
    print(f"Muon param shapes: {muon_param_shapes}")

    if dist.is_initialized():
        optimisers = [
            DistAdam(adam_params, **train_cfg.adam_cfg.__dict__),
            DistAdam(muon_params, **train_cfg.muon_cfg.__dict__),
        ]
    else:
        optimisers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg.adam_cfg.lr,
                betas=train_cfg.adam_cfg.betas,
                eps=train_cfg.adam_cfg.eps,
                weight_decay=train_cfg.adam_cfg.weight_decay,
            )
        ]

    if model.yoco:
        L = len(model.blocks) // 2  # 2 blocks per layer because of HC
        window_sizes = [model.window_size] * (L // 2) + [None] * (L - L // 2)
    elif model.window_size is None:
        window_sizes = None
    else:
        ws = model.window_size
        sws = ws // 2
        window_sizes = [None, ws, sws, sws, ws, sws, sws, None, sws, sws, sws, ws]
    print(f"{window_sizes=}")

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
                window_sizes=window_sizes,
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
            if i == train_cfg.grad_accum_steps - 1:
                for optimiser in optimisers:
                    if isinstance(optimiser, DistAdam):
                        optimiser.should_sync = True
            inputs, targets, seqlens = next(train_loader)
            loss = model(inputs, targets, seqlens, window_sizes)
            loss = loss / train_cfg.grad_accum_steps
            batch_loss += (
                loss.detach()
                / train_cfg.train_batch_tokens
                * train_cfg.grad_accum_steps
                * world_size
            )
            loss.backward()
        for optimiser in optimisers:
            if isinstance(optimiser, DistAdam):
                optimiser.should_sync = False
                if optimiser.ortho_fn is None and step % 2 == 0:
                    continue  # skip every other step for adamw
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
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

    from torch.nn import ReLU

    from modded_nanogpt.gpt import GPTConfig, ReLU2, RMSNorm
    from modded_nanogpt.opt import get_lr_schedule, newtonschulz5
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

    def print0(s, console=False):
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

        # reduces train/val steps, 1 = full training
        DEBUG_FACTOR = 512 if not is_cuda(device) else 8

        # reduces mini batch size and sequence length (if > 16),
        # increases grad accum steps to keep tokens per batch constant
        VRAM_FACTOR = 16 if not is_cuda(device) else 4

        GRAD_ACCUM_STEPS = 8 * VRAM_FACTOR
        MINI_BATCH_SIZE = max(1, 16 // VRAM_FACTOR)
        MAX_SEQ_LEN = 2048 if MINI_BATCH_SIZE > 1 else 2048 * 16 // VRAM_FACTOR

        TRAIN_STEPS = max(1, 2245 // DEBUG_FACTOR)

        TRAIN_BATCH_TOKENS = MAX_SEQ_LEN * MINI_BATCH_SIZE * GRAD_ACCUM_STEPS
        # VAL_BATCH_TOKENS = TRAIN_BATCH_TOKENS
        VAL_BATCH_TOKENS = 2_097_152 // VRAM_FACTOR // 2

        lr_schedule_fn = partial(
            get_lr_schedule, num_steps=TRAIN_STEPS * DEBUG_FACTOR, cooldown_frac=0.5
        )
        # lr_schedule_fn = None
        train_cfg = TrainConfig(
            # data
            train_files="data/fineweb10B/fineweb_train_*.bin",
            val_files="data/fineweb10B/fineweb_val_*.bin",
            train_batch_tokens=TRAIN_BATCH_TOKENS,
            train_max_seq_len=MAX_SEQ_LEN,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            val_tokens=10_485_760 // DEBUG_FACTOR,
            val_batch_tokens=VAL_BATCH_TOKENS,
            # optimisation
            num_steps=TRAIN_STEPS,
            adam_cfg=OptimConfig(
                lr=8e-3,
                betas=(0.65, 0.95),
                eps=1e-8,
                weight_decay=0.0,
                ortho_fn=None,
                lr_schedule_fn=lr_schedule_fn,
            ),
            muon_cfg=OptimConfig(
                lr=3e-2,
                betas=(0.95, 0.95),
                eps=1e-10,
                weight_decay=0.0,
                ortho_fn=newtonschulz5,
                lr_schedule_fn=lr_schedule_fn,
            ),
            # eval and logging
            val_steps=max(1, TRAIN_STEPS // 20),  # 0 for only at end
            vals_per_ckpt=0,  # 0 for only at end
            use_wandb=False and master_process,
        )

        model_cfg = GPTConfig(
            vocab_size=next_multiple(50_257, 128),  # 50_304
            num_layers=12,
            num_heads=6,
            dim=768,
            max_seq_len=max(
                MAX_SEQ_LEN, VAL_BATCH_TOKENS // (GRAD_ACCUM_STEPS * world_size)
            ),
            norm=partial(RMSNorm, elementwise_affine=False),
            rope=True,
            qk_norm=True,
            act=ReLU2,
            bf16=True,
            hc=True,
            dynamic=True,
            expansion_rate=1,
            dnorm=partial(RMSNorm, elementwise_affine=False),
            window_size=256,
            shc_lr_mul=1.0,
            dhc_lr_mul=1.0,
            kernel_options={
                "BLOCK_M": 128 // VRAM_FACTOR,
                "BLOCK_N": 128 // VRAM_FACTOR,
                "BLOCK_M1": 64 // VRAM_FACTOR,
                "BLOCK_N1": 128 // VRAM_FACTOR,
                "BLOCK_M2": 128 // VRAM_FACTOR,
                "BLOCK_N2": 64 // VRAM_FACTOR,
            },
            yoco=False,
        )
        model = GPT(model_cfg).to(device)

        print0("Compiling model...")
        model = torch.compile(model, dynamic=True, fullgraph=False)
        print0("Model compiled.")

        if dist.is_initialized():
            for param in model.parameters():
                dist.broadcast(param.detach(), src=0)

        print0(f"{device=}")
        print0(
            f"VRAM_FACTOR={VRAM_FACTOR} GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} MINI_BATCH_SIZE={MINI_BATCH_SIZE}"
            f" MAX_SEQ_LEN={MAX_SEQ_LEN} TRAIN_BATCH_TOKENS={TRAIN_BATCH_TOKENS} TRAIN_TOKENS={TRAIN_BATCH_TOKENS * train_cfg.num_steps}"
        )
        print0(model_cfg.__dict__ | train_cfg.__dict__)
        print0(model)

        train(model, train_cfg, device)
        print0("Training complete.")

    except Exception:
        if master_process:
            print0("An exception occurred during training:")
            print0(traceback.format_exc())
        sys.exit(1)

    finally:
        print(logfile)
        if dist.is_initialized():
            dist.destroy_process_group()

        max_memory_used = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if is_cuda(device)
            else 0.0
        )
        print0(f"Max memory used: {max_memory_used:.2f} GB")

        torch.save(model.state_dict(), "final.pt")
