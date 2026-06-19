"""Exponential moving average of model weights — reused from V7 unchanged.

Standard practice in SOTA single-image and stereo SR since 2022. The EMA copy
of the weights tracks a slower moving average; validation runs on EMA. Typical
gain: +0.05–0.15 dB on Flickr1024 ×4 free.

Decay schedule
--------------
Linearly ramp the effective decay from 0 (EMA = live weights) toward the target
``decay`` over ``warmup_steps`` iterations, equivalent to "uniform weighting of
all steps seen so far." This prevents the EMA from being dominated by the random
initialisation at iter 100.

DDP
---
EMA updates run on the raw unwrapped model. Since DDP synchronises params after
``optimizer.step()``, all ranks compute identical EMA updates — no all-reduce
needed for the EMA itself.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain an exponential moving average of model parameters and buffers.

    Usage
    -----
    ```
    ema = ModelEMA(raw_model, decay=0.9999, warmup_steps=2000)
    for step in range(total_iters):
        optimizer.step()
        ema.update(raw_model)
    val_psnr = validate(ema.module, val_loader)
    ```
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        device: torch.device | str | None = None,
    ):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0

        self.module = copy.deepcopy(model).eval()
        if device is not None:
            self.module.to(device)
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _effective_decay(self) -> float:
        """Effective decay this step, ramped linearly from 0 to ``decay`` over
        ``warmup_steps`` updates.

        Early on the EMA tracks the (fast-moving) model closely; the cap by
        ``self.decay`` means that once ``num_updates >= warmup_steps`` the
        effective decay is exactly ``self.decay``.
        """
        if self.warmup_steps <= 0:
            return self.decay
        warm = self.num_updates / self.warmup_steps
        return min(self.decay, warm)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the EMA copy. Call AFTER optimizer.step()."""
        decay = self._effective_decay()
        ema_params = dict(self.module.named_parameters())
        for name, p in model.named_parameters():
            if name not in ema_params:
                continue
            ema_p = ema_params[name]
            if not p.requires_grad:
                ema_p.data.copy_(p.data)
            else:
                ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        ema_buffers = dict(self.module.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].data.copy_(b.data)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "module": self.module.state_dict(),
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd["module"])
        self.decay = sd.get("decay", self.decay)
        self.warmup_steps = sd.get("warmup_steps", self.warmup_steps)
        self.num_updates = sd.get("num_updates", 0)
