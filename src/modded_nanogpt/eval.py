import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from modded_nanogpt.data import distributed_data_generator
from modded_nanogpt.gpt import GPT
from modded_nanogpt.util import is_cuda, is_mps


class Clock:
    def __init__(self, device: torch.device | str):
        self.elapsed_ms = 0.0
        if is_cuda(device):
            self.sync_fn = torch.cuda.synchronize
        elif is_mps(device):
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


@torch.no_grad()
def eval(
    model: GPT,
    filename_pattern: str,
    val_tokens: int,
    batch_tokens: int,
    max_seq_len: int,
    grad_accum_steps: int,
    device: str,
) -> torch.Tensor:
    model.eval()

    val_loss = 0.0
    val_loader = distributed_data_generator(
        filename_pattern=filename_pattern,
        batch_tokens=batch_tokens,
        max_seq_len=max_seq_len,
        grad_accum_steps=grad_accum_steps,
        device=device,
    )

    # 1 val step = 1 mini batch (vs 1 train step = grad_accum_steps mini batches)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    val_steps = grad_accum_steps * val_tokens // batch_tokens // world_size
    for _ in tqdm(range(val_steps), desc="Validation", total=val_steps, leave=False):
        inputs, targets = next(val_loader)
        _, loss = model(inputs, targets)
        val_loss += loss
    val_loss /= val_steps

    model.train()
    return val_loss
