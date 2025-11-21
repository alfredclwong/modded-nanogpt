import time

import torch
from tqdm import tqdm

from modded_nanogpt.data import distributed_data_generator
from modded_nanogpt.gpt import GPT


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


@torch.no_grad()
def eval(
    model: GPT,
    filename_pattern: str,
    val_tokens: int,
    batch_tokens: int,
    max_seq_len: int,
    grad_accum_steps: int,
    device: str,
) -> float:
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
    val_steps = grad_accum_steps * val_tokens // batch_tokens
    for _ in tqdm(range(val_steps), desc="Validation", total=val_steps, leave=False):
        inputs, targets = next(val_loader)
        _, loss = model(inputs, targets)
        val_loss += loss.item()
    val_loss /= val_steps

    model.train()
    return val_loss
