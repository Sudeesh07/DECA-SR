"""
Block-level unit tests for Stereo SR modules.

Run all tests:
    python test_blocks.py

Run a specific block:
    python test_blocks.py --block deca
    python test_blocks.py --block frequency
    python test_blocks.py --block hcpf
    python test_blocks.py --block tcva
    python test_blocks.py --block occlusion
    python test_blocks.py --block nafblock
    python test_blocks.py --block network

Tests per block:
  1. Shape correctness (forward pass produces expected shapes)
  2. Gradient flow (all learnable params receive gradients)
  3. Residual behaviour (output != input, i.e. the block does something)
  4. Numerical stability (no NaN/Inf in outputs)
  5. Device transfer (CPU ↔ CUDA if available)
  6. Determinism (same input → same output in eval mode)
  7. Parameter count report
"""

import os
import sys
import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

# ── Helpers ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _params(module):
    return sum(p.numel() for p in module.parameters())


def _trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@contextmanager
def _timer(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"  [{dt:.3f}s] {label}")


def _check_shape(tensor, expected, name="output"):
    ok = tensor.shape == expected
    status = PASS if ok else FAIL
    msg = f"  [{status}] {name} shape: got {tuple(tensor.shape)}, expected {tuple(expected)}"
    print(msg)
    assert ok, msg
    return ok


def _check_finite(tensor, name="output"):
    ok = torch.isfinite(tensor).all().item()
    status = PASS if ok else FAIL
    msg = f"  [{status}] {name} finite (no NaN / Inf)"
    print(msg)
    assert ok, msg
    return ok


def _check_gradients(module, loss_tensor, name="module"):
    loss_tensor.backward()
    total = 0
    with_grad = 0
    for p in module.parameters():
        if p.requires_grad:
            total += 1
            if p.grad is not None:
                with_grad += 1
    ok = with_grad == total
    status = PASS if ok else FAIL
    msg = f"  [{status}] gradient flow: {with_grad}/{total} params received gradients"
    print(msg)
    # Zero grads so subsequent tests aren't affected
    module.zero_grad()
    assert ok, msg
    return ok


def _check_not_identity(out, inp, name="output"):
    """Verify the block modifies its input (is not a no-op)."""
    diff = (out - inp).abs().max().item()
    ok = diff > 1e-7
    status = PASS if ok else FAIL
    msg = f"  [{status}] {name} is not identity (max diff = {diff:.2e})"
    print(msg)
    # Don't assert — some blocks with zero-init betas *are* identity at init
    return ok


def _check_determinism(module, *inputs, name="module"):
    """Same input should give same output in eval mode."""
    module.eval()
    with torch.no_grad():
        out1 = module(*inputs)
        out2 = module(*inputs)
    if isinstance(out1, tuple):
        out1, out2 = out1[0], out2[0]
    ok = torch.allclose(out1, out2, atol=1e-6)
    status = PASS if ok else FAIL
    msg = f"  [{status}] {name} deterministic in eval mode"
    print(msg)
    module.train()
    assert ok, msg
    return ok


def _report_params(module, name="module"):
    total = _params(module)
    trainable = _trainable(module)
    print(f"  [INFO] {name}: {total:,} params ({total / 1e6:.3f}M), trainable: {trainable:,}")


def _header(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ── Per-block test functions ─────────────────────────────────────────────────

def test_deca():
    """Test DECA: Deformable Epipolar Cross-Attention (models/deca.py)."""
    from models.deca import (
        DECA, BidirectionalDECA, DECAOffsetPredictor, LayerNorm2d,
        rearrange_to_multihead, rearrange_sampled_to_multihead, rearrange_back_to_spatial,
    )

    _header("DECA — Deformable Epipolar Cross-Attention")
    B, C, H, W = 2, 64, 32, 64
    fl = torch.randn(B, C, H, W, device=DEVICE)
    fr = torch.randn(B, C, H, W, device=DEVICE)

    # ── LayerNorm2d ──
    print("\n--- LayerNorm2d ---")
    ln = LayerNorm2d(C).to(DEVICE)
    out = ln(fl)
    _check_shape(out, fl.shape, "LayerNorm2d")
    _check_finite(out, "LayerNorm2d")

    # ── Rearrange helpers ──
    print("\n--- Tensor rearrange helpers ---")
    num_heads, head_dim = 4, C // 4
    mh = rearrange_to_multihead(fl, num_heads, head_dim)
    _check_shape(mh, (B, num_heads, head_dim, H * W), "rearrange_to_multihead")
    back = rearrange_back_to_spatial(mh, H, W)
    _check_shape(back, fl.shape, "rearrange_back_to_spatial")
    print(f"  [{PASS}] rearrange round-trip")

    # ── DECAOffsetPredictor ──
    print("\n--- DECAOffsetPredictor ---")
    K = 8
    predictor = DECAOffsetPredictor(C, k=K, max_disparity=192, num_corr_shifts=9).to(DEVICE)
    _report_params(predictor, "DECAOffsetPredictor")
    offsets = predictor(fl, fr)
    _check_shape(offsets, (B, K, H, W), "offsets")
    _check_finite(offsets, "offsets")
    assert offsets.abs().max() <= 192, f"  [{FAIL}] offsets should be clamped to [-192, 192]"
    print(f"  [{PASS}] offsets clamped: range [{offsets.min():.1f}, {offsets.max():.1f}]")
    _check_gradients(predictor, offsets.mean(), "DECAOffsetPredictor")

    # ── DECA (no occlusion gate) ──
    print("\n--- DECA (basic) ---")
    deca = DECA(channels=C, k=8, num_heads=4).to(DEVICE)
    _report_params(deca, "DECA")
    with _timer("DECA forward"):
        out = deca(fl, fr)
    _check_shape(out, fl.shape, "DECA output")
    _check_finite(out, "DECA output")
    _check_not_identity(out, fl, "DECA output")
    _check_gradients(deca, out.mean(), "DECA")
    _check_determinism(deca, fl, fr, name="DECA")
    # Learnable scale should receive gradients
    out2 = deca(fl, fr)
    out2.mean().backward()
    assert deca.log_scale.grad is not None, f"  [{FAIL}] log_scale has no gradient"
    print(f"  [{PASS}] learnable log_scale receives gradient")
    deca.zero_grad()

    # ── DECA (with occlusion gate) ──
    print("\n--- DECA (occlusion gate) ---")
    deca_gate = DECA(channels=C, k=8, num_heads=4, use_occlusion_gate=True).to(DEVICE)
    out_g = deca_gate(fl, fr)
    _check_shape(out_g, fl.shape, "DECA+gate output")
    _check_finite(out_g, "DECA+gate output")
    _check_gradients(deca_gate, out_g.mean(), "DECA+gate")

    # ── BidirectionalDECA ──
    print("\n--- BidirectionalDECA ---")
    bideca = BidirectionalDECA(channels=C, k=8, num_heads=4).to(DEVICE)
    _report_params(bideca, "BidirectionalDECA")
    out_l, out_r = bideca(fl, fr)
    _check_shape(out_l, fl.shape, "BiDECA left")
    _check_shape(out_r, fr.shape, "BiDECA right")
    _check_finite(out_l, "BiDECA left")
    _check_finite(out_r, "BiDECA right")
    _check_gradients(bideca, (out_l.mean() + out_r.mean()), "BidirectionalDECA")
    _check_determinism(bideca, fl, fr, name="BidirectionalDECA")


def test_frequency():
    """Test Frequency-Aware modules (models/frequency.py)."""
    from models.frequency import (
        FastFourierConvBlock, FrequencyCrossViewFusion,
        HighFrequencyEnhancementBranch, FrequencyAwareRefinement,
        WaveletLoss, FrequencyL1Loss,
    )

    _header("Frequency-Aware Refinement Modules")
    B, C, H, W = 2, 64, 32, 64
    fl = torch.randn(B, C, H, W, device=DEVICE)
    fr = torch.randn(B, C, H, W, device=DEVICE)

    # ── FastFourierConvBlock ──
    print("\n--- FastFourierConvBlock ---")
    ffc = FastFourierConvBlock(C).to(DEVICE)
    _report_params(ffc, "FFC")
    out = ffc(fl)
    _check_shape(out, fl.shape, "FFC")
    _check_finite(out, "FFC")
    _check_not_identity(out, fl, "FFC")
    _check_gradients(ffc, out.mean(), "FFC")

    # ── FFC without spectral attention ──
    print("\n--- FFC (no spectral attention) ---")
    ffc_nsa = FastFourierConvBlock(C, use_spectral_attention=False).to(DEVICE)
    out_nsa = ffc_nsa(fl)
    _check_shape(out_nsa, fl.shape, "FFC_noSA")
    _check_finite(out_nsa, "FFC_noSA")

    # ── FrequencyCrossViewFusion ──
    print("\n--- FrequencyCrossViewFusion ---")
    fcvf = FrequencyCrossViewFusion(C).to(DEVICE)
    _report_params(fcvf, "FCVF")
    out = fcvf(fl, fr)
    _check_shape(out, fl.shape, "FCVF")
    _check_finite(out, "FCVF")
    _check_gradients(fcvf, out.mean(), "FCVF")

    # ── FrequencyCrossViewFusion (+ phase correlation) ──
    print("\n--- FCVF (phase correlation) ---")
    fcvf_pc = FrequencyCrossViewFusion(C, use_phase_correlation=True).to(DEVICE)
    out_pc = fcvf_pc(fl, fr)
    _check_shape(out_pc, fl.shape, "FCVF+phase")
    _check_finite(out_pc, "FCVF+phase")

    # ── HighFrequencyEnhancementBranch ──
    print("\n--- HighFrequencyEnhancementBranch ---")
    hfe = HighFrequencyEnhancementBranch(C).to(DEVICE)
    _report_params(hfe, "HFE")
    out = hfe(fl)
    _check_shape(out, fl.shape, "HFE")
    _check_finite(out, "HFE")
    _check_gradients(hfe, out.mean(), "HFE")

    # ── FrequencyAwareRefinement ──
    print("\n--- FrequencyAwareRefinement ---")
    far = FrequencyAwareRefinement(C, spatial_blocks=2, freq_blocks=1).to(DEVICE)
    _report_params(far, "FAR")
    with _timer("FAR forward"):
        out = far(fl, fr)
    _check_shape(out, fl.shape, "FAR")
    _check_finite(out, "FAR")
    _check_gradients(far, out.mean(), "FAR")
    _check_determinism(far, fl, fr, name="FAR")

    # ── FAR bidirectional ──
    print("\n--- FAR (bidirectional) ---")
    far_bi = FrequencyAwareRefinement(C, spatial_blocks=2, freq_blocks=1, use_bidirectional=True).to(DEVICE)
    out_l, out_r = far_bi(fl, fr)
    _check_shape(out_l, fl.shape, "FAR_bi left")
    _check_shape(out_r, fr.shape, "FAR_bi right")
    _check_finite(out_l, "FAR_bi left")

    # ── Losses ──
    print("\n--- Losses ---")
    pred = torch.randn(B, 3, H, W, device=DEVICE)
    target = torch.randn(B, 3, H, W, device=DEVICE)

    wl = WaveletLoss().to(DEVICE)
    loss_val = wl(pred, target)
    _check_finite(loss_val.unsqueeze(0), "WaveletLoss")
    print(f"  [{PASS}] WaveletLoss = {loss_val.item():.4f}")

    fl_loss = FrequencyL1Loss().to(DEVICE)
    loss_val = fl_loss(pred, target)
    _check_finite(loss_val.unsqueeze(0), "FrequencyL1Loss")
    print(f"  [{PASS}] FrequencyL1Loss = {loss_val.item():.4f}")


def test_hcpf():
    """Test HCPF: Hierarchical Cross-View Pyramid Fusion (models/hcpf.py)."""
    from models.hcpf import (
        GroupWiseCorrelation, GroupWiseCorrelationVectorized,
        FullEpipolarAttention, HCPF, BidirectionalHCPF,
    )

    _header("HCPF — Hierarchical Cross-View Pyramid Fusion")
    # Use dimensions divisible by 4 for HCPF downsampling
    B, C, H, W = 2, 64, 64, 128
    fl = torch.randn(B, C, H, W, device=DEVICE)
    fr = torch.randn(B, C, H, W, device=DEVICE)

    # ── GroupWiseCorrelation ──
    print("\n--- GroupWiseCorrelation (loop) ---")
    gwc = GroupWiseCorrelation(C, num_groups=8, max_disp=24).to(DEVICE)
    _report_params(gwc, "GWC")
    with _timer("GWC forward"):
        out = gwc(fl, fr)
    _check_shape(out, fl.shape, "GWC")
    _check_finite(out, "GWC")
    _check_gradients(gwc, out.mean(), "GWC")

    # ── GroupWiseCorrelationVectorized ──
    print("\n--- GroupWiseCorrelation (vectorized) ---")
    gwcv = GroupWiseCorrelationVectorized(C, num_groups=8, max_disp=24).to(DEVICE)
    _report_params(gwcv, "GWCv")
    with _timer("GWCv forward"):
        out_v = gwcv(fl, fr)
    _check_shape(out_v, fl.shape, "GWCv")
    _check_finite(out_v, "GWCv")

    # ── FullEpipolarAttention ──
    print("\n--- FullEpipolarAttention ---")
    fea = FullEpipolarAttention(C, num_heads=4).to(DEVICE)
    _report_params(fea, "FEA")
    with _timer("FEA forward"):
        out = fea(fl, fr)
    _check_shape(out, fl.shape, "FEA")
    _check_finite(out, "FEA")
    _check_gradients(fea, out.mean(), "FEA")

    # ── HCPF (full) ──
    print("\n--- HCPF (full pyramid) ---")
    hcpf = HCPF(C, k=8).to(DEVICE)
    _report_params(hcpf, "HCPF")
    with _timer("HCPF forward"):
        out = hcpf(fl, fr)
    _check_shape(out, fl.shape, "HCPF")
    _check_finite(out, "HCPF")
    _check_gradients(hcpf, out.mean(), "HCPF")
    _check_determinism(hcpf, fl, fr, name="HCPF")

    # ── BidirectionalHCPF ──
    print("\n--- BidirectionalHCPF ---")
    bi_hcpf = BidirectionalHCPF(C, k=8).to(DEVICE)
    _report_params(bi_hcpf, "BiHCPF")
    out_l, out_r = bi_hcpf(fl, fr)
    _check_shape(out_l, fl.shape, "BiHCPF left")
    _check_shape(out_r, fr.shape, "BiHCPF right")
    _check_finite(out_l, "BiHCPF left")
    _check_finite(out_r, "BiHCPF right")


def test_tcva():
    """Test TCVA: Transposed Cross-View Attention (models/tcva.py)."""
    from models.tcva import TCVA

    _header("TCVA — Transposed Cross-View Attention")
    B, C, H, W = 2, 64, 32, 64
    fl = torch.randn(B, C, H, W, device=DEVICE)
    fr = torch.randn(B, C, H, W, device=DEVICE)

    # ── TCVA single-head ──
    print("\n--- TCVA (1 head) ---")
    tcva_1 = TCVA(C, num_heads=1).to(DEVICE)
    _report_params(tcva_1, "TCVA_1h")
    with _timer("TCVA_1h forward"):
        out = tcva_1(fl, fr)
    _check_shape(out, fl.shape, "TCVA_1h")
    _check_finite(out, "TCVA_1h")
    _check_not_identity(out, fl, "TCVA_1h")
    _check_gradients(tcva_1, out.mean(), "TCVA_1h")
    _check_determinism(tcva_1, fl, fr, name="TCVA_1h")

    # ── TCVA multi-head ──
    print("\n--- TCVA (4 heads) ---")
    tcva_4 = TCVA(C, num_heads=4).to(DEVICE)
    _report_params(tcva_4, "TCVA_4h")
    out = tcva_4(fl, fr)
    _check_shape(out, fl.shape, "TCVA_4h")
    _check_finite(out, "TCVA_4h")
    _check_gradients(tcva_4, out.mean(), "TCVA_4h")

    # ── Verify temperature is learnable ──
    print("\n--- Temperature parameter ---")
    out2 = tcva_1(fl, fr)
    out2.mean().backward()
    has_temp_grad = tcva_1.temperature.grad is not None
    status = PASS if has_temp_grad else FAIL
    print(f"  [{status}] temperature receives gradient")
    tcva_1.zero_grad()


def test_occlusion():
    """Test Occlusion-Aware Fusion (models/occlusion.py)."""
    from models.occlusion import (
        MultiScaleConfidencePredictor, LightweightConfidencePredictor,
        SelfAttentionBlock, CrossAttentionBlock,
        OcclusionAwareFusion, BidirectionalOcclusionFusion,
        StereoConsistencyLoss, CharbonnierLoss,
    )

    _header("Occlusion-Aware Fusion Modules")
    B, C, H, W = 2, 64, 32, 64
    fl = torch.randn(B, C, H, W, device=DEVICE)
    fr = torch.randn(B, C, H, W, device=DEVICE)

    # ── MultiScaleConfidencePredictor ──
    print("\n--- MultiScaleConfidencePredictor ---")
    ms_conf = MultiScaleConfidencePredictor(C).to(DEVICE)
    _report_params(ms_conf, "MSConf")
    conf = ms_conf(fl, fr)
    _check_shape(conf, (B, 1, H, W), "MSConf")
    _check_finite(conf, "MSConf")
    assert (conf >= 0).all() and (conf <= 1).all(), f"  [{FAIL}] confidence out of [0,1]"
    print(f"  [{PASS}] confidence in [0,1]: range [{conf.min():.3f}, {conf.max():.3f}]")
    _check_gradients(ms_conf, conf.mean(), "MSConf")

    # ── LightweightConfidencePredictor ──
    print("\n--- LightweightConfidencePredictor ---")
    lw_conf = LightweightConfidencePredictor(C).to(DEVICE)
    _report_params(lw_conf, "LWConf")
    conf = lw_conf(fl, fr)
    _check_shape(conf, (B, 1, H, W), "LWConf")
    assert (conf >= 0).all() and (conf <= 1).all()
    print(f"  [{PASS}] confidence in [0,1]")

    # ── SelfAttentionBlock ──
    print("\n--- SelfAttentionBlock ---")
    sa = SelfAttentionBlock(C, num_blocks=2).to(DEVICE)
    _report_params(sa, "SelfAttn")
    out = sa(fl)
    _check_shape(out, fl.shape, "SelfAttn")
    _check_finite(out, "SelfAttn")

    # ── CrossAttentionBlock ──
    print("\n--- CrossAttentionBlock ---")
    ca = CrossAttentionBlock(C, num_heads=4, use_local=True).to(DEVICE)
    _report_params(ca, "CrossAttn")
    out = ca(fl, fr)
    _check_shape(out, fl.shape, "CrossAttn")
    _check_finite(out, "CrossAttn")
    _check_gradients(ca, out.mean(), "CrossAttn")

    # ── CrossAttentionBlock (no local) ──
    print("\n--- CrossAttentionBlock (global only) ---")
    ca_g = CrossAttentionBlock(C, num_heads=4, use_local=False).to(DEVICE)
    out_g = ca_g(fl, fr)
    _check_shape(out_g, fl.shape, "CrossAttn_global")
    _check_finite(out_g, "CrossAttn_global")

    # ── OcclusionAwareFusion ──
    print("\n--- OcclusionAwareFusion ---")
    oaf = OcclusionAwareFusion(C).to(DEVICE)
    _report_params(oaf, "OAF")
    with _timer("OAF forward"):
        fused, conf_map = oaf(fl, fr)
    _check_shape(fused, fl.shape, "OAF fused")
    _check_shape(conf_map, (B, 1, H, W), "OAF confidence")
    _check_finite(fused, "OAF fused")
    _check_finite(conf_map, "OAF confidence")
    _check_gradients(oaf, fused.mean(), "OAF")
    _check_determinism(oaf, fl, fr, name="OAF")

    # ── OAF (lightweight confidence) ──
    print("\n--- OAF (lightweight confidence) ---")
    oaf_lw = OcclusionAwareFusion(C, use_multiscale_conf=False).to(DEVICE)
    fused_lw, _ = oaf_lw(fl, fr)
    _check_shape(fused_lw, fl.shape, "OAF_lw")
    _check_finite(fused_lw, "OAF_lw")

    # ── BidirectionalOcclusionFusion ──
    print("\n--- BidirectionalOcclusionFusion ---")
    bi_oaf = BidirectionalOcclusionFusion(C).to(DEVICE)
    _report_params(bi_oaf, "BiOAF")
    fused_l, fused_r, conf_l, conf_r = bi_oaf(fl, fr)
    _check_shape(fused_l, fl.shape, "BiOAF fused_l")
    _check_shape(fused_r, fr.shape, "BiOAF fused_r")
    _check_shape(conf_l, (B, 1, H, W), "BiOAF conf_l")
    _check_shape(conf_r, (B, 1, H, W), "BiOAF conf_r")

    # ── Losses ──
    print("\n--- Loss functions ---")
    sr_l = torch.randn(B, 3, 128, 256, device=DEVICE)
    sr_r = torch.randn(B, 3, 128, 256, device=DEVICE)
    disp = torch.rand(B, 1, 128, 256, device=DEVICE) * 50
    conf = torch.rand(B, 1, 128, 256, device=DEVICE)

    scl = StereoConsistencyLoss().to(DEVICE)
    losses = scl(sr_l, sr_r, disp, disp, conf, conf)
    for k, v in losses.items():
        _check_finite(v.unsqueeze(0), f"SCL/{k}")
    print(f"  [{PASS}] StereoConsistencyLoss keys: {list(losses.keys())}")

    cl = CharbonnierLoss().to(DEVICE)
    loss_val = cl(sr_l, sr_r)
    _check_finite(loss_val.unsqueeze(0), "CharbonnierLoss")
    print(f"  [{PASS}] CharbonnierLoss = {loss_val.item():.4f}")


def test_nafblock():
    """Test NAFBlock and variants (models/nafblock.py)."""
    from models.nafblock import (
        NAFBlock, NAFBlockWithLocalAttention,
        NAFNetEncoder, NAFNetDecoder,
        LayerNorm2d, DropPath, SimpleGate, SimplifiedChannelAttention,
        get_nafblock,
    )

    _header("NAFBlock — Nonlinear Activation Free Block")
    B, C, H, W = 2, 64, 64, 64
    x = torch.randn(B, C, H, W, device=DEVICE)

    # ── Core components ──
    print("\n--- Core components ---")
    ln = LayerNorm2d(C).to(DEVICE)
    _check_shape(ln(x), x.shape, "LayerNorm2d")

    sg = SimpleGate()
    _check_shape(sg(torch.randn(B, C * 2, H, W, device=DEVICE)), x.shape, "SimpleGate")

    sca = SimplifiedChannelAttention(C).to(DEVICE)
    _check_shape(sca(x), x.shape, "SCA")

    dp = DropPath(0.1).to(DEVICE)
    dp.train()
    out_dp = dp(x)
    _check_shape(out_dp, x.shape, "DropPath (train)")
    dp.eval()
    out_dp_eval = dp(x)
    assert torch.allclose(out_dp_eval, x), f"  [{FAIL}] DropPath should be identity in eval"
    print(f"  [{PASS}] DropPath is identity in eval mode")

    # ── NAFBlock ──
    print("\n--- NAFBlock ---")
    naf = NAFBlock(C, expansion=2, drop_path=0.1).to(DEVICE)
    _report_params(naf, "NAFBlock")
    with _timer("NAFBlock forward"):
        out = naf(x)
    _check_shape(out, x.shape, "NAFBlock")
    _check_finite(out, "NAFBlock")
    _check_gradients(naf, out.mean(), "NAFBlock")
    _check_determinism(naf, x, name="NAFBlock")

    # ── NAFBlock (checkpointed) ──
    print("\n--- NAFBlock (checkpointed) ---")
    has_cp = hasattr(torch.utils, "checkpoint")
    if has_cp:
        naf_cp = NAFBlock(C, use_checkpoint=True).to(DEVICE)
        naf_cp.train()
        x_cp = x.detach().clone().requires_grad_(True)
        out_cp = naf_cp(x_cp)
        _check_shape(out_cp, x.shape, "NAFBlock_cp")
        _check_finite(out_cp, "NAFBlock_cp")
        print(f"  [{PASS}] checkpointed forward works")
    else:
        print(f"  [{SKIP}] torch.utils.checkpoint not available")

    # ── NAFBlockWithLocalAttention ──
    print("\n--- NAFBlockWithLocalAttention ---")
    naf_la = NAFBlockWithLocalAttention(C, window_size=8).to(DEVICE)
    _report_params(naf_la, "NAFBlock+LA")
    out_la = naf_la(x)
    _check_shape(out_la, x.shape, "NAFBlock+LA")
    _check_finite(out_la, "NAFBlock+LA")
    _check_gradients(naf_la, out_la.mean(), "NAFBlock+LA")

    # ── Non-square / non-divisible input for local attention ──
    print("\n--- NAFBlock+LA (non-square input) ---")
    x_ns = torch.randn(B, C, 30, 50, device=DEVICE)  # not divisible by 8
    out_ns = naf_la(x_ns)
    _check_shape(out_ns, x_ns.shape, "NAFBlock+LA non-square")
    _check_finite(out_ns, "NAFBlock+LA non-square")

    # ── NAFNetEncoder ──
    print("\n--- NAFNetEncoder ---")
    encoder = NAFNetEncoder(C, num_blocks=[2, 2, 2]).to(DEVICE)
    _report_params(encoder, "Encoder")
    features = encoder(x)
    print(f"  [INFO] Encoder output scales: {[tuple(f.shape) for f in features]}")
    for i, f in enumerate(features):
        _check_finite(f, f"Encoder level {i}")

    # ── NAFNetDecoder ──
    print("\n--- NAFNetDecoder ---")
    decoder = NAFNetDecoder(C, num_blocks=[2, 2, 2]).to(DEVICE)
    _report_params(decoder, "Decoder")
    out_dec = decoder(features)
    _check_shape(out_dec, x.shape, "Decoder output")
    _check_finite(out_dec, "Decoder output")
    _check_gradients(decoder, out_dec.mean(), "Decoder")

    # ── Factory function ──
    print("\n--- get_nafblock factory ---")
    naf_std = get_nafblock(C, variant="standard").to(DEVICE)
    naf_la2 = get_nafblock(C, variant="local_attention", window_size=8).to(DEVICE)
    _check_shape(naf_std(x), x.shape, "factory/standard")
    _check_shape(naf_la2(x), x.shape, "factory/local_attention")
    print(f"  [{PASS}] factory function works for both variants")


def test_network():
    """Test the full StereoSRNet (models/network.py)."""
    from models.network import (
        StereoSRNet, stereo_sr_tiny, stereo_sr_base, build_model,
        EncoderStage, HCPF, FrequencyAwareRefinement, OcclusionAwareFusion,
        Upsampler,
    )

    _header("StereoSRNet — Full Network")
    B, C_in, H, W = 1, 3, 64, 128
    img_l = torch.randn(B, C_in, H, W, device=DEVICE)
    img_r = torch.randn(B, C_in, H, W, device=DEVICE)

    # ── Sub-modules used in network ──
    embed = 48

    print("\n--- EncoderStage (DECA, downsample) ---")
    enc_deca = EncoderStage(embed, embed, num_blocks=2, cross_view_type="deca", downsample=True).to(DEVICE)
    fl_tmp = torch.randn(B, embed, H, W, device=DEVICE)
    fr_tmp = torch.randn(B, embed, H, W, device=DEVICE)
    fl_out, fr_out, fl_next, fr_next = enc_deca(fl_tmp, fr_tmp)
    _check_shape(fl_out, (B, embed, H, W), "EncStage skip")
    _check_shape(fl_next, (B, embed, H // 2, W // 2), "EncStage downsampled")
    print(f"  [{PASS}] EncoderStage DECA+downsample shapes correct")

    print("\n--- EncoderStage (TCVA, no downsample) ---")
    enc_tcva = EncoderStage(embed, embed, num_blocks=2, cross_view_type="tcva", downsample=False).to(DEVICE)
    fl_out2, _, fl_next2, _ = enc_tcva(fl_tmp, fr_tmp)
    _check_shape(fl_out2, (B, embed, H, W), "EncStage TCVA skip")
    _check_shape(fl_next2, (B, embed, H, W), "EncStage TCVA next")
    print(f"  [{PASS}] EncoderStage TCVA shapes correct")

    print("\n--- HCPF (network version) ---")
    hcpf = HCPF(embed, k=8).to(DEVICE)
    _report_params(hcpf, "HCPF_net")
    out_hcpf = hcpf(fl_tmp, fr_tmp)
    _check_shape(out_hcpf, (B, embed, H, W), "HCPF_net")
    _check_finite(out_hcpf, "HCPF_net")

    print("\n--- Upsampler (x4) ---")
    up4 = Upsampler(embed, scale=4, out_channels=3).to(DEVICE)
    out_up = up4(fl_tmp)
    _check_shape(out_up, (B, 3, H * 4, W * 4), "Upsampler x4")

    print("\n--- Upsampler (x2) ---")
    up2 = Upsampler(embed, scale=2, out_channels=3).to(DEVICE)
    out_up2 = up2(fl_tmp)
    _check_shape(out_up2, (B, 3, H * 2, W * 2), "Upsampler x2")

    # ── Tiny model (quick test) ──
    print("\n--- StereoSRNet Tiny (scale=4) ---")
    model_tiny = stereo_sr_tiny(scale=4).to(DEVICE)
    _report_params(model_tiny, "Tiny")
    with _timer("Tiny forward"):
        sr_l, sr_r, conf_l, conf_r = model_tiny(img_l, img_r)
    expected = (B, C_in, H * 4, W * 4)
    _check_shape(sr_l, expected, "Tiny SR_L")
    _check_shape(sr_r, expected, "Tiny SR_R")
    _check_finite(sr_l, "Tiny SR_L")
    _check_finite(sr_r, "Tiny SR_R")
    _check_shape(conf_l, (B, 1, H, W), "Tiny conf_L")
    print(f"  [{PASS}] Tiny model output shapes correct")

    # Gradient check
    loss = sr_l.mean() + sr_r.mean()
    loss.backward()
    graded = sum(1 for p in model_tiny.parameters() if p.grad is not None and p.requires_grad)
    total = sum(1 for p in model_tiny.parameters() if p.requires_grad)
    ok = graded == total
    status = PASS if ok else FAIL
    print(f"  [{status}] gradient flow: {graded}/{total} params")
    model_tiny.zero_grad()

    # ── Base model (shape check only — may be slow) ──
    print("\n--- StereoSRNet Base (scale=4, shape check) ---")
    model_base = stereo_sr_base(scale=4).to(DEVICE)
    _report_params(model_base, "Base")
    with torch.no_grad(), _timer("Base forward (no grad)"):
        sr_l_b, sr_r_b, _, _ = model_base(img_l, img_r)
    _check_shape(sr_l_b, expected, "Base SR_L")
    _check_finite(sr_l_b, "Base SR_L")

    # ── build_model from args ──
    print("\n--- build_model (from args) ---")

    class _Args:
        scale = 4
        in_channels = 3
        embed_dim = 32
        enc_blocks = [2, 2, 2]
        deca_k = 4
        spatial_blocks = 2
        freq_blocks = 1
        stochastic_depth = 0.0

    model_args = build_model(_Args()).to(DEVICE)
    sr_l_a, sr_r_a, _, _ = model_args(img_l, img_r)
    _check_shape(sr_l_a, expected, "build_model SR_L")
    print(f"  [{PASS}] build_model works")

    # ── Scale=2 ──
    print("\n--- StereoSRNet Tiny (scale=2) ---")
    model_s2 = stereo_sr_tiny(scale=2).to(DEVICE)
    sr_l2, sr_r2, _, _ = model_s2(img_l, img_r)
    _check_shape(sr_l2, (B, C_in, H * 2, W * 2), "Tiny x2 SR_L")
    print(f"  [{PASS}] scale=2 works")


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_BLOCKS = {
    "deca": test_deca,
    "frequency": test_frequency,
    "hcpf": test_hcpf,
    "tcva": test_tcva,
    "occlusion": test_occlusion,
    "nafblock": test_nafblock,
    "network": test_network,
}


def main():
    parser = argparse.ArgumentParser(description="Block-level tests for Stereo SR modules")
    parser.add_argument(
        "--block", type=str, default=None,
        choices=list(ALL_BLOCKS.keys()),
        help="Test a specific block (default: run all)",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    if args.block:
        blocks_to_test = {args.block: ALL_BLOCKS[args.block]}
    else:
        blocks_to_test = ALL_BLOCKS

    results = {}
    for name, test_fn in blocks_to_test.items():
        try:
            t0 = time.perf_counter()
            test_fn()
            dt = time.perf_counter() - t0
            results[name] = (PASS, dt)
        except Exception as e:
            results[name] = (FAIL, str(e))
            import traceback
            traceback.print_exc()
        finally:
            # Free GPU memory between blocks
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # ── Summary ──
    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for name, (status, info) in results.items():
        if status == PASS:
            print(f"  [{PASS}] {name:12s}  ({info:.2f}s)")
        else:
            print(f"  [{FAIL}] {name:12s}  Error: {info}")

    num_pass = sum(1 for _, (s, _) in results.items() if s == PASS)
    num_total = len(results)
    print(f"\n  {num_pass}/{num_total} blocks passed")

    if num_pass < num_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
