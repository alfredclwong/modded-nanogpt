from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    ortho_fn: Callable[[torch.Tensor], torch.Tensor] | None


@torch.compile(dynamic=False, fullgraph=True)
def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class DistAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        ortho_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # init state
        for p in params:
            # Store original size for params that need padding
            original_size = p.size(0)
            padded_size = self._get_padded_size(original_size)
            chunk_size = padded_size // self.world_size

            exp_avg = torch.zeros_like(
                p[:chunk_size], dtype=torch.bfloat16, device=p[0].device
            )
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(
                step=0,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                original_size=original_size,
                padded_size=padded_size,
            )
        # DistributedAdam implementation by @vagrawal, @akash5474

        self.should_sync = False

        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        self.register_backward_hooks()

        self.ortho_fn = ortho_fn

    def _get_padded_size(self, size: int) -> int:
        """Calculate padded size to make it divisible by world_size."""
        remainder = size % self.world_size
        if remainder == 0:
            return size
        return size + (self.world_size - remainder)

    def register_backward_hooks(self):
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            for param in params:
                hook = param.register_post_accumulate_grad_hook(self._sync_gradient)
                self._reduce_scatter_hooks.append(hook)

    @torch.compile
    @torch.no_grad()
    def _sync_gradient(self, param):
        if not self.should_sync:
            return

        grad = param.grad
        state = self.state[param]
        original_size = state["original_size"]
        padded_size = state["padded_size"]

        # Pad gradient if necessary
        if original_size != padded_size:
            grad_padded = torch.zeros(
                (padded_size,) + grad.shape[1:], dtype=grad.dtype, device=grad.device
            )
            grad_padded[:original_size] = grad
            grad = grad_padded

        # Now gradient is always divisible by world_size
        rank_size = padded_size // self.world_size
        grad_slice = torch.empty_like(grad[:rank_size])
        self._reduce_scatter_futures[param] = (
            dist.reduce_scatter_tensor(
                grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
            ).get_future(),
            grad_slice,
        )

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        all_gather_futures: list[torch.Future] = []

        for group in reversed(self.param_groups):
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for param in reversed(group["params"]):
                if param not in self._reduce_scatter_futures:
                    continue

                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()

                state = self.state[param]
                original_size = state["original_size"]
                padded_size = state["padded_size"]
                rank_size = padded_size // self.world_size

                # Create padded parameter view if necessary
                if original_size != padded_size:
                    param_padded = torch.zeros(
                        (padded_size,) + param.shape[1:],
                        dtype=param.dtype,
                        device=param.device,
                    )
                    param_padded[:original_size] = param
                else:
                    param_padded = param

                p_slice = param_padded[rank * rank_size : (rank + 1) * rank_size]
                lr = group["lr"] * getattr(param, "lr_mul", 1.0)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(exp_avg, exp_avg, value=1 - beta2)
                # orthogonalisation step
                if self.ortho_fn is not None:
                    exp_avg = self.ortho_fn(exp_avg)
                # bias corrections
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2**0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)

                all_gather_futures.append(
                    dist.all_gather_into_tensor(
                        param_padded, p_slice, async_op=True
                    ).get_future()
                )

                # If parameter was padded, we'll need to copy back only the original portion
                if original_size != padded_size:
                    state["param_padded"] = param_padded

        # Wait for all futures
        torch.futures.collect_all(all_gather_futures).wait()

        # Copy back unpadded results
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if "param_padded" in state:
                    param.copy_(state["param_padded"][: state["original_size"]])
                    del state["param_padded"]

        self._reduce_scatter_futures.clear()
