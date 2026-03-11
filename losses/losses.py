import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features[:26]))
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.features(x)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat = self.vgg(pred)
        with torch.no_grad():
            target_feat = self.vgg(target)
        return F.l1_loss(pred_feat, target_feat)

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, nf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RelativisticAverageGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward_d(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        real_label = torch.ones_like(d_real)
        fake_label = torch.zeros_like(d_fake)
        d_loss = (self.bce(d_real - d_fake.mean(), real_label) +
                  self.bce(d_fake - d_real.mean(), fake_label)) / 2
        return d_loss

    def forward_g(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        real_label = torch.ones_like(d_real)
        fake_label = torch.zeros_like(d_fake)
        g_loss = (self.bce(d_real - d_fake.mean(), fake_label) +
                  self.bce(d_fake - d_real.mean(), real_label)) / 2
        return g_loss

class WaveletSubbandLoss(nn.Module):
    def __init__(self, wavelet: str = "haar", level: int = 2):
        super().__init__()
        self.level = level
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.register_buffer("filters", torch.stack([ll, lh, hl, hh]).unsqueeze(1))

    def dwt2(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        subbands = F.conv2d(x, self.filters, stride=2, padding=0)
        subbands = subbands.reshape(B, C, 4, H // 2, W // 2)
        ll = subbands[:, :, 0]
        lh = subbands[:, :, 1]
        hl = subbands[:, :, 2]
        hh = subbands[:, :, 3]
        return ll, (lh, hl, hh)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        p, t = pred, target

        for lv in range(self.level):
            p_ll, (p_lh, p_hl, p_hh) = self.dwt2(p)
            t_ll, (t_lh, t_hl, t_hh) = self.dwt2(t)
            weight = 2.0 ** (self.level - lv)
            loss += weight * (F.l1_loss(p_lh, t_lh) +
                              F.l1_loss(p_hl, t_hl) +
                              F.l1_loss(p_hh, t_hh))
            loss += F.l1_loss(p_ll, t_ll)
            p, t = p_ll, t_ll

        return loss

class PhotometricStereoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sr_left: torch.Tensor, sr_right: torch.Tensor,
                conf_left=None, conf_right=None) -> torch.Tensor:
        B, C, H, W = sr_left.shape
        loss = 0.0
        count = 0
        for d in [1, 2, 4, 8, 16]:
            if d >= W:
                continue
            diff = (sr_left[:, :, :, d:] - sr_right[:, :, :, :W-d]).abs()

            if conf_left is not None:
                w = conf_left[:, :, :, d:]
                loss += (diff * w).mean()
            else:
                loss += diff.mean()
            count += 1

        return loss / max(count, 1)

class CombinedLoss(nn.Module):
    def __init__(self, w_pixel=1.0, w_percep=0.1, w_adv=0.01,
                 w_freq=0.3, w_stereo=0.2, device="cuda"):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_percep = w_percep
        self.w_adv = w_adv
        self.w_freq = w_freq
        self.w_stereo = w_stereo

        self.pixel_loss = CharbonnierLoss()
        self.percep_loss = PerceptualLoss().to(device) if w_percep > 0 else None
        self.freq_loss = WaveletSubbandLoss().to(device)
        self.stereo_loss = PhotometricStereoLoss()

        self.use_percep = False
        self.use_adv = False
        self.use_freq = False
        self.use_stereo = False

    def set_phase(self, phase: int):
        if phase >= 1:
            pass
        if phase >= 2:
            self.use_freq = True
            self.use_stereo = True
        if phase >= 3:
            self.use_percep = True
        if phase >= 4:
            self.use_adv = True

    def forward(self, sr_left, sr_right, gt_left, gt_right,
                conf_left=None, conf_right=None):
        losses = {}
        l_pixel = (self.pixel_loss(sr_left, gt_left) +
                   self.pixel_loss(sr_right, gt_right)) / 2
        losses["pixel"] = l_pixel

        total = self.w_pixel * l_pixel
        if self.use_freq:
            l_freq = (self.freq_loss(sr_left, gt_left) +
                      self.freq_loss(sr_right, gt_right)) / 2
            losses["freq"] = l_freq
            total += self.w_freq * l_freq
        if self.use_stereo:
            l_stereo = self.stereo_loss(sr_left, sr_right, conf_left, conf_right)
            losses["stereo"] = l_stereo
            total += self.w_stereo * l_stereo
        if self.use_percep and self.percep_loss is not None:
            l_percep = (self.percep_loss(sr_left, gt_left) +
                        self.percep_loss(sr_right, gt_right)) / 2
            losses["percep"] = l_percep
            total += self.w_percep * l_percep
        if conf_left is not None:
            eps = 1e-7
            entropy_l = -(conf_left * torch.log(conf_left + eps) +
                          (1 - conf_left) * torch.log(1 - conf_left + eps))
            entropy_r = -(conf_right * torch.log(conf_right + eps) +
                          (1 - conf_right) * torch.log(1 - conf_right + eps))
            l_conf = (entropy_l.mean() + entropy_r.mean()) / 2
            losses["conf_reg"] = l_conf
            total += 0.05 * l_conf

        losses["total"] = total
        return total, losses
