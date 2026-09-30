from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1):
        defaults = {"lr":lr, "beta1": betas[0], "beta2": betas[1], "t":1, "epsilon": eps, "lambda_": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        for group in self.param_groups:
            a = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            t = group["t"]
            a_t = a * math.sqrt(1-beta2 ** t)/(1-beta1**t)
            ep = group["epsilon"]
            lambda_ =  group["lambda_"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                g = p.grad.data
                # In-place updates: fewer kernel launches and no temporaries,
                # matching the original update equations exactly.
                p.data.mul_(1 - a * lambda_)
                m = state.get("m")
                if m is None:
                    m = state["m"] = torch.zeros_like(p)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v = state.get("v")
                if v is None:
                    v = state["v"] = torch.zeros_like(p)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = v.sqrt().add_(ep)
                p.data.addcdiv_(m, denom, value=-a_t)
            group["t"] = t+1


