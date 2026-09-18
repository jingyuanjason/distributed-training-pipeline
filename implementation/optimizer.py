from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1):
        defaults = {"lr":lr, "beta1": betas[0], "beta2": betas[1], "t":1, "epsilon": eps, "lambda_": weight_decay}
        super().__init__(params, defaults)

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
                p.data -= a * lambda_ * p.data
                m = state.get("m", 0)
                m = beta1 * m  + (1 - beta1) * g
                state["m"] = m
                v = state.get("v", 0)
                v = beta2 * v + (1 - beta2) * (g ** 2)
                state["v"] = v
                p.data -= a_t * m / (torch.sqrt(v) + ep)
            group["t"] = t+1


