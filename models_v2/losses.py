"""
Loss functions for Stereo SR V2 training.

Multi-phase curriculum:
  Phase 1: Pixel (Charbonnier) only
  Phase 2: + Wavelet frequency + Stereo photometric
  Phase 3: + Perceptual (VGG) + Confidence entropy regularization
  Phase 4: + Adversarial (RaGAN) — GAN loss computed via forward_g / forward_d
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Pixel Loss ──────────────────────────────────────────────────────────


class CharbonnierLoss(nn.Module):
    """sqrt(diff^2 + eps^2) — smoother than L1, more robust than L2."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


# ── Perceptual Loss ─────────────────────────────────────────────────────


class VGGFeatureExtractor(nn.Module):
    """VGG19 relu4_4 features with ImageNet normalization."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:26])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features((x - self.mean) / self.std)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat = self.vgg(pred)
        with torch.no_grad():
            target_feat = self.vgg(target)
        return F.l1_loss(pred_feat, target_feat)


# ── Adversarial Loss ────────────────────────────────────────────────────


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, nf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 1, 1), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 2, 4, 2, 1), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 4, 2, 1), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(nf * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelativisticAverageGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward_d(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self.bce(d_real - d_fake.mean(), torch.ones_like(d_real))
            + self.bce(d_fake - d_real.mean(), torch.zeros_like(d_fake))
        )

    def forward_g(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self.bce(d_real - d_fake.mean(), torch.zeros_like(d_real))
            + self.bce(d_fake - d_real.mean(), torch.ones_like(d_fake))
        )


# ── Frequency Loss ──────────────────────────────────────────────────────


class WaveletSubbandLoss(nn.Module):
    """Multi-level Haar DWT loss emphasizing high-frequency subbands."""

    def __init__(self, level: int = 2):
        super().__init__()
        self.level = level
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.register_buffer("filters", torch.stack([ll, lh, hl, hh]).unsqueeze(1))

    def _dwt2(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        subbands = F.conv2d(x, self.filters, stride=2)
        subbands = subbands.reshape(b, c, 4, h // 2, w // 2)
        return subbands[:, :, 0], (subbands[:, :, 1], subbands[:, :, 2], subbands[:, :, 3])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pred.new_tensor(0.0)
        p, t = pred, target
        for lv in range(self.level):
            p_ll, (p_lh, p_hl, p_hh) = self._dwt2(p)
            t_ll, (t_lh, t_hl, t_hh) = self._dwt2(t)
            weight = 2.0 ** (self.level - lv)
            loss = loss + weight * (F.l1_loss(p_lh, t_lh) + F.l1_loss(p_hl, t_hl) + F.l1_loss(p_hh, t_hh))
            loss = loss + F.l1_loss(p_ll, t_ll)
            p, t = p_ll, t_ll
        return loss


# ── Stereo Consistency Loss ─────────────────────────────────────────────


class PhotometricStereoLoss(nn.Module):
    """Symmetric shift-based stereo consistency."""

    def __init__(self, disparities: Tuple[int, ...] = (1, 2, 4, 8, 16)):
        super().__init__()
        self.disparities = disparities

    @staticmethod
    def _resize_conf(conf: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if conf.shape[-2:] == (h, w):
            return conf
        return F.interpolate(conf, size=(h, w), mode="bilinear", align_corners=False)

    def _one_direction(self, src, tgt, conf):
        _, _, h, w = src.shape
        total = src.new_tensor(0.0)
        count = 0
        conf_r = self._resize_conf(conf, h, w) if conf is not None else None
        for d in self.disparities:
            if d >= w:
                continue
            diff = (src[:, :, :, d:] - tgt[:, :, :, :w - d]).abs()
            if conf_r is not None:
                total = total + (diff * conf_r[:, :, :, d:]).mean()
            else:
                total = total + diff.mean()
            count += 1
        return total / max(count, 1)

    def forward(self, sr_left, sr_right, conf_left=None, conf_right=None):
        l2r = self._one_direction(sr_left, sr_right, conf_left)
        r2l = self._one_direction(sr_right, sr_left, conf_right)
        return 0.5 * (l2r + r2l)


# ── Combined Multi-Phase Loss ───────────────────────────────────────────


class CombinedLoss(nn.Module):
    """
    Multi-phase loss with curriculum activation.

    Phase 1: pixel only
    Phase 2: + frequency + stereo
    Phase 3: + perceptual + confidence entropy regularization
    Phase 4: + adversarial (GAN loss added via compute_gan_g_loss, discriminator
              updated separately in training loop via compute_gan_d_loss)

    Adversarial note:
        GAN training requires a separate discriminator and optimizer. This module
        provides compute_gan_g_loss() and compute_gan_d_loss() helpers that the
        training loop should call when phase >= 4. The generator loss is added to
        the total; the discriminator loss is returned separately for its own
        optimizer step.
    """

    def __init__(
        self,
        w_pixel: float = 1.0,
        w_percep: float = 0.1,
        w_adv: float = 0.01,
        w_freq: float = 0.3,
        w_stereo: float = 0.2,
        w_conf_reg: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_percep = w_percep
        self.w_adv = w_adv
        self.w_freq = w_freq
        self.w_stereo = w_stereo
        self.w_conf_reg = w_conf_reg

        self.pixel_loss = CharbonnierLoss()
        self.percep_loss = PerceptualLoss().to(device) if w_percep > 0 else None
        self.freq_loss = WaveletSubbandLoss().to(device)
        self.stereo_loss = PhotometricStereoLoss()
        self.gan_loss = RelativisticAverageGANLoss()

        self.use_percep = False
        self.use_adv = False
        self.use_freq = False
        self.use_stereo = False
        self.use_conf_reg = False

    def set_phase(self, phase: int):
        self.use_freq = phase >= 2
        self.use_stereo = phase >= 2
        self.use_percep = phase >= 3
        self.use_conf_reg = phase >= 3
        self.use_adv = phase >= 4

    def compute_gan_g_loss(
        self,
        discriminator: nn.Module,
        sr: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generator adversarial loss for one view.
        Call this in the training loop when phase >= 4, after the main forward.

        Args:
            discriminator: the discriminator module
            sr: super-resolved output [B, 3, H, W]
            gt: ground truth [B, 3, H, W]

        Returns:
            Weighted generator GAN loss (ready to add to total)
        """
        d_fake = discriminator(sr)
        with torch.no_grad():
            d_real = discriminator(gt)
        return self.w_adv * self.gan_loss.forward_g(d_real, d_fake)

    def compute_gan_d_loss(
        self,
        discriminator: nn.Module,
        sr: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator loss for one view.
        Call this in the training loop for the discriminator optimizer step.

        Args:
            discriminator: the discriminator module
            sr: super-resolved output [B, 3, H, W] (detached)
            gt: ground truth [B, 3, H, W]

        Returns:
            Discriminator loss
        """
        d_real = discriminator(gt)
        d_fake = discriminator(sr.detach())
        return self.gan_loss.forward_d(d_real, d_fake)

    def forward(
        self,
        sr_left: torch.Tensor,
        sr_right: torch.Tensor,
        gt_left: torch.Tensor,
        gt_right: torch.Tensor,
        conf_left: Optional[torch.Tensor] = None,
        conf_right: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}

        l_pixel = 0.5 * (self.pixel_loss(sr_left, gt_left) + self.pixel_loss(sr_right, gt_right))
        losses["pixel"] = l_pixel
        total = self.w_pixel * l_pixel

        if self.use_freq:
            l_freq = 0.5 * (self.freq_loss(sr_left, gt_left) + self.freq_loss(sr_right, gt_right))
            losses["freq"] = l_freq
            total = total + self.w_freq * l_freq

        if self.use_stereo:
            l_stereo = self.stereo_loss(sr_left, sr_right, conf_left, conf_right)
            losses["stereo"] = l_stereo
            total = total + self.w_stereo * l_stereo

        if self.use_percep and self.percep_loss is not None:
            l_percep = 0.5 * (self.percep_loss(sr_left, gt_left) + self.percep_loss(sr_right, gt_right))
            losses["percep"] = l_percep
            total = total + self.w_percep * l_percep

        # Confidence entropy regularization: pushes C toward 0 or 1 (decisive).
        # Only active from phase 3 onward, after the network has learned meaningful
        # stereo correspondences. Running this in phase 1 would lock randomly
        # initialized confidence into premature binary patterns.
        if self.use_conf_reg and conf_left is not None and conf_right is not None:
            cl = F.interpolate(conf_left, size=sr_left.shape[-2:], mode="bilinear", align_corners=False)
            cr = F.interpolate(conf_right, size=sr_right.shape[-2:], mode="bilinear", align_corners=False)
            eps = 1e-7
            ent_l = -(cl * torch.log(cl + eps) + (1 - cl) * torch.log(1 - cl + eps))
            ent_r = -(cr * torch.log(cr + eps) + (1 - cr) * torch.log(1 - cr + eps))
            l_conf = 0.5 * (ent_l.mean() + ent_r.mean())
            losses["conf_reg"] = l_conf
            total = total + self.w_conf_reg * l_conf

        # Note: adversarial loss is NOT computed here because it requires a
        # discriminator module and separate optimizer steps. Use
        # compute_gan_g_loss() and compute_gan_d_loss() in the training loop
        # when self.use_adv is True (phase >= 4).

        losses["total"] = total
        return total, losses