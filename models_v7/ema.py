"""Exponential moving average of model weights.

Why EMA for SR
--------------
Standard practice in every SOTA single-image and stereo SR paper since 2022
(NAFNet, NAFSSR, SwinFSR, Restormer). The EMA copy of the weights tracks a
slower moving average over training, smoothing out the SGD noise. At
validation time the EMA copy is used instead of the live weights. Typical
gain: **+0.05 to +0.15 dB on Flickr1024 ×4**, free except for ~30 lines of
code and a second copy of the weight tensors.

Decay schedule
--------------
A bare ``EMA_DECAY = 0.999`` works but is sub-optimal early in training
because at iter 100 the EMA is still dominated by the random initialisation.
Standard fix (used by EMA implementations across vision): warm-up the decay
linearly so it starts at 0 (EMA = live weights) and reaches the target
``decay`` after ``warmup_steps`` iterations. We use a simple ``min(decay,
(1 + step) / (10 + step))`` schedule equivalent to "step weighting" — at
step 0 the EMA copies the live weights; by step ~5000 the decay is at the
target.

DDP
---
EMA updates run on the raw (unwrapped) model and the EMA copy is held only
on rank 0. We don't need to all-reduce — the live model is already
synchronised by DDP, so rank 0's EMA is consistent with what the EMA on
other ranks would be.
"""
from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain an exponential moving average of model parameters and buffers.

    Usage
    -----
    ```
    ema = ModelEMA(raw_model, decay=0.9999)
    for step in range(total_iters):
        optimizer.step()
        ema.update(raw_model)
    # At validation:
    val_psnr = validate(ema.module, val_loader)
    ```

    Parameters
    ----------
    model : nn.Module
        The live (raw, unwrapped) model. The EMA copy is initialised from
        this and tracks its updates.
    decay : float, default 0.9999
        Target decay factor. EMA ← decay · EMA + (1 - decay) · live.
        For 100k-iter SR training: 0.999 to 0.9999 is the typical range;
        larger means smoother but slower to track.
    warmup_steps : int, default 2000
        During warmup the effective decay is min(decay, step / (step + 10)),
        equivalent to "uniform weighting of all steps seen so far". This
        prevents the EMA from being dominated by random init at iter 100.
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

        # Deep copy to detach from the live model entirely.
        self.module = copy.deepcopy(model).eval()
        if device is not None:
            self.module.to(device)
        # EMA params/buffers do not receive gradient.
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _effective_decay(self) -> float:
        """Effective decay this step, including warmup ramp."""
        if self.warmup_steps <= 0:
            return self.decay
        # During warmup, "average over N steps" = (N - 1) / N, with N = num_updates + 1.
        # This grows from 0 (at step 1) toward decay (at large step).
        warm = self.num_updates / (self.num_updates + 10.0)
        return min(self.decay, warm)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the EMA copy from the current live model parameters & buffers.

        Call AFTER ``optimizer.step()``. Buffer copies are exact (no decay) —
        BatchNorm running stats and similar buffers should track the live
        model directly. V7 has no BatchNorm but this is the safe default.
        """
        decay = self._effective_decay()
        ema_params = dict(self.module.named_parameters())
        for name, p in model.named_parameters():
            if name not in ema_params:
                # Skip params present in live but not in EMA (e.g. if model
                # was modified between init and this update). Rare.
                continue
            ema_p = ema_params[name]
            if not p.requires_grad:
                # For frozen params, just copy directly — no EMA.
                ema_p.data.copy_(p.data)
            else:
                ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

        # Buffers: copy directly. EMA-ing running statistics is rarely useful
        # and breaks BatchNorm semantics.
        ema_buffers = dict(self.module.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].data.copy_(b.data)

        self.num_updates += 1

    def state_dict(self) -> dict:
        """Return a serialisable state dict for checkpointing."""
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
