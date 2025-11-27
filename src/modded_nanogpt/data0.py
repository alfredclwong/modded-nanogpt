import glob
import threading
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor

from modded_nanogpt.util import is_cuda, next_multiple_of_n


def _load_data_shard(file: Path, device):
    header = torch.from_file(
        str(file), False, 256, dtype=torch.int32
    )  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=is_cuda(device)
        )  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(
            memoryview(tokens.cpu().numpy()).cast("B")
        )  # zero-copy readinto
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


BOS_ID = 50256


class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens = tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (
                (tokens[:4_000_000] == BOS_ID)
                .nonzero(as_tuple=True)[0]
                .to(torch.int64)
                .cpu()
                .numpy()
            )
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (
                (tokens == BOS_ID)
                .nonzero(as_tuple=True)[0]
                .to(torch.int64)
                .cpu()
                .numpy()
            )
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (
            (self.tokens == BOS_ID)
            .nonzero(as_tuple=True)[0]
            .to(torch.int64)
            .cpu()
            .numpy()
        )
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        # if quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter == 5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(
                        f"Insufficient BOS ahead of position {cur}; hit tail of shard."
                    )
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(
                    self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                    cur + max_seq_len,
                    cur + num_tokens_local - cur_len + 1,
                )
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter += 1
        return starts, ends


class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
    def __init__(self, file_iter, device, world_size: int = 1):
        self.file_iter = file_iter
        self.device = device
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter), self.device)
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data


def distributed_data_generator(
    *,
    filename_pattern: str,
    num_tokens: int,
    max_seq_len: int,
    grad_accum_steps: int,
    align_to_bos: bool,
    device: str | torch.device,
):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, (
        "Batch size must be divisible by world size"
    )
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter), device)
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, device, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(
            num_tokens_local // 300, n=128
        )  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = (
                    torch.tensor(seq_starts[rank]),
                    torch.tensor(seq_ends[rank]),
                )
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= (
                1  # last document was too long to account for _targets offset
            )
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter), device), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local : pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(
                num_tokens_local,
            )
            _targets = buf[1:].view(
                num_tokens_local,
            )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1 : len(cum_lengths) + 1] = cum_lengths

        _inputs = _inputs.unsqueeze(0)

        non_blocking = is_cuda(device)
        new_params = yield (
            _inputs.to(device=device, dtype=torch.int32, non_blocking=non_blocking),
            _targets.to(device=device, dtype=torch.int64, non_blocking=non_blocking),
            _cum_lengths.to(
                device=device, dtype=torch.int32, non_blocking=non_blocking
            ),
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, (
                "Num tokens must be divisible by world size"
            )
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps
