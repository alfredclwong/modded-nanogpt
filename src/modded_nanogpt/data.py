"""
- Each fineweb data shard is 200MB = 100 million tokens (uint16)
- There are 103 training shards and 1 validation shard in fineweb10B
- The original llm.c code required 10B tokens for training, SOTA <0.6B (README needs update)
- Start with simple data loading without prefetching, distributed io, or bos alignment
"""

from pathlib import Path

import torch
import torch.distributed as dist

from modded_nanogpt.util import is_cuda


def _load_data_shard(file: Path, device: torch.device | str) -> torch.Tensor:
    """Load a single data shard from disk into a uint16 tensor."""
    header = torch.from_file(str(file), shared=False, size=256, dtype=torch.int32)
    assert header[0] == 20240520, f"magic number mismatch in {file}"
    assert header[1] == 1, f"version mismatch in {file}"
    num_tokens = int(header[2])
    with open(file, "rb", buffering=0) as f:  # disable buffering for raw io
        f.seek(256 * 4)  # skip header
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=is_cuda(device)
        )  # pre-allocate
        num_bytes = f.readinto(memoryview(tokens.numpy()).cast("B"))  # zero-copy
        assert num_bytes == num_tokens * 2, (
            f"read {num_bytes} bytes, expected {num_tokens * 2}"
        )
    return tokens


def distributed_data_generator(
    filename_pattern: str,
    batch_tokens: int,
    max_seq_len: int,
    grad_accum_steps: int,
    device: torch.device | str,
):
    """Yield mini-batches of (inputs, targets) tensors for training.

    Args:
        filename_pattern: glob pattern to match data shard files
        batch_tokens: total number of tokens per batch (across grad accum steps)
        max_seq_len: maximum sequence length for the model
        grad_accum_steps: number of gradient accumulation steps per update

    Yields:
        inputs: Tensor of shape [mini_batch_size, max_seq_len], dtype torch.int32
        targets: Tensor of shape [mini_batch_size, max_seq_len], dtype torch.int64
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert batch_tokens % (world_size * grad_accum_steps * max_seq_len) == 0, (
        f"{batch_tokens=} % ({world_size=} * {grad_accum_steps=} * {max_seq_len=}) != 0"
    )

    # mini-batches split the batch tokens across grad accum steps and data parallel workers
    mini_batch_tokens = batch_tokens // grad_accum_steps // world_size
    mini_batch_size = mini_batch_tokens // max_seq_len

    files = sorted(Path().glob(filename_pattern))
    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter), device)
    pos = 0

    while True:
        # load next shard if needed
        if pos + mini_batch_tokens + 1 >= len(tokens):  # +1 for target shift
            try:
                # note: could try to keep leftover tokens but competition rules forbid changes
                tokens = _load_data_shard(next(file_iter), device)
            except StopIteration:
                break
            pos = 0

        # extract mini-batch
        start = pos + rank * mini_batch_tokens
        inputs = tokens[start : start + mini_batch_tokens]
        targets = tokens[start + 1 : start + mini_batch_tokens + 1]
        inputs = inputs.view(mini_batch_size, max_seq_len)
        targets = targets.view(mini_batch_size, max_seq_len)
        pos += mini_batch_tokens * world_size

        yield (
            inputs.to(
                device=device, dtype=torch.int32, non_blocking=is_cuda(device)
            ),
            targets.to(
                device=device, dtype=torch.int64, non_blocking=is_cuda(device)
            ),
        )
