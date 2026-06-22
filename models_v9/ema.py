"""Exponential moving average of model weights (V9, copied from the fixed V8).

Linearly ramps the effective decay 0 -> ``decay`` over ``warmup_steps``, then
holds at ``decay``. Validation runs on the EMA copy. DDP-safe (EMA updates run
on the raw model after the optimizer step, identical across ranks).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
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
        if self.warmup_steps <= 0:
            return self.decay
        warm = self.num_updates / self.warmup_steps
        return min(self.decay, warm)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
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

    def _effective_decay_value(self) -> float:
        return self._effective_decay()

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
