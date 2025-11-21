import torch


def is_cuda(device: torch.device | str) -> bool:
    return str(device).startswith("cuda")


def is_mps(device: torch.device | str) -> bool:
    return str(device).startswith("mps")


def next_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple
