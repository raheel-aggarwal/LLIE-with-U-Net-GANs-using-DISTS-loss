#!/usr/bin/env python3
"""
inference.py — Attention U-Net GAN inference for SID Sony low-light RAW
enhancement.

Standalone by design (duplicates the small pieces it needs from
training.py — raw packing, ratio parsing, the Generator class, sRGB
caching — so this file runs on its own given just a trained model + a data
folder).

Two ways to run it:

  1) Batch mode over a split list (default: the test list from config.yaml):
       python inference.py --model_path ./models/models/best \\
           --config config.yaml --output_dir ./results

  2) Single-file mode (one-off .ARW, no data_root/list file needed):
       python inference.py --model_path ./models/models/best \\
           --input /path/to/00001_00_0.04s.ARW --ratio 250 \\
           --output_dir ./results
       # (--gt /path/to/long.ARW instead of --ratio also works, and enables --eval)

Output naming: every input `<name>.ARW` is saved as `test_<name>.png` in
--output_dir (exposure string and all, extension swapped to .png).

The sRGB cache (`short_rgb/`, `long_rgb/` under data_root) is reused if
present and built on the fly if not — the same cache training.py writes to,
so a from-scratch inference run over the test split doesn't require
re-running training.py first.
"""
import os
import glob
import argparse

import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import rawpy
except ImportError:
    rawpy = None

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None


def get_lpips_model(device='cpu'):
    """Load the LPIPS Alex model from a local cache under $MODEL_DIR if possible.

    The lpips package may try to download its pretrained weights on first use.
    This helper checks a local cache directory first and reuses it on subsequent
    runs, avoiding repeated downloads.
    """
    if lpips_lib is None:
        print("[eval] lpips not installed — skipping LPIPS, PSNR/SSIM still computed.")
        return None

    model_dir = os.environ.get('MODEL_DIR')
    if not model_dir:
        print("[eval] MODEL_DIR not set — using lpips' default download behavior.")
        return lpips_lib.LPIPS(net='alex').to(device)

    cache_dir = os.path.join(model_dir, 'lpips')
    os.makedirs(cache_dir, exist_ok=True)

    # lpips stores its weights under a model-specific folder; prefer a stable
    # local path if the expected file exists, otherwise let lpips initialize.
    expected_files = [
        os.path.join(cache_dir, 'weights.pth'),
        os.path.join(cache_dir, 'v0.1', 'alex.pth'),
        os.path.join(cache_dir, 'alex.pth'),
    ]
    if any(os.path.isfile(path) for path in expected_files):
        print(f"[eval] using cached LPIPS weights from {cache_dir}")
        return lpips_lib.LPIPS(net='alex', model_path=cache_dir).to(device)

    try:
        print(f"[eval] downloading LPIPS weights to {cache_dir}")
        model = lpips_lib.LPIPS(net='alex').to(device)
        print(f"[eval] LPIPS model ready from {cache_dir}")
        return model
    except Exception as e:
        print(f"[eval] failed to initialize LPIPS from {cache_dir}: {e}")
        return None

try:
    from huggingface_hub import PyTorchModelHubMixin
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

    class PyTorchModelHubMixin:
        pass


# =============================================================================
# RAW <-> tensor helpers (identical to training.py — kept standalone here)
# =============================================================================
def pack_raw_array(raw_visible, black_level=512, white_level=16383):
    im = raw_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level - black_level)
    im = np.expand_dims(im, axis=2)
    H, W, _ = im.shape
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                           im[0:H:2, 1:W:2, :],
                           im[1:H:2, 1:W:2, :],
                           im[1:H:2, 0:W:2, :]), axis=2)
    return out


def pack_raw_file(path, black_level=512, white_level=16383):
    if rawpy is None:
        raise RuntimeError("rawpy is required to read .ARW files but is not installed.")
    with rawpy.imread(path) as raw:
        arr = raw.raw_image_visible.copy()
    return pack_raw_array(arr, black_level, white_level)


def demosaic_to_srgb_u16(raw_path):
    if rawpy is None:
        raise RuntimeError("rawpy is required to build the RGB cache but is not installed.")
    with rawpy.imread(raw_path) as raw:
        im = raw.postprocess(use_camera_wb=True, half_size=False,
                              no_auto_bright=True, output_bps=16)
    return im


def parse_exposure(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    tok = stem.split('_')[-1]
    return float(tok[:-1]) if tok.endswith('s') else float(tok)


def parse_scene_id(filename):
    return int(os.path.basename(filename).split('_')[0])


def compute_ratio(short_path, long_path, cap=300.0):
    return min(parse_exposure(long_path) / parse_exposure(short_path), cap)


def read_list_file(list_path):
    pairs = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            pairs.append((os.path.basename(parts[0]), os.path.basename(parts[1])))
    return pairs


def _atomic_imwrite(path, img):
    tmp_path = path + f".tmp{os.getpid()}.png"
    if not cv2.imwrite(tmp_path, img):
        raise RuntimeError(f"cv2.imwrite failed for {tmp_path}")
    os.replace(tmp_path, path)


def _atomic_save_npy(path, arr):
    tmp_path = path + f".tmp{os.getpid()}"
    np.save(tmp_path, arr)
    os.replace(tmp_path + ".npy", path)


def ensure_rgb_cache(raw_path, rgb_dir, black_level=512, white_level=16383, is_short=False, packed_dir=None):
    """Make sure the sRGB PNG cache for `raw_path` exists under `rgb_dir`
    (same basename, .png). Builds it if missing. If `is_short` and
    `packed_dir` is given, also ensures the packed .npy cache. Returns
    (rgb_path, packed_path_or_None)."""
    os.makedirs(rgb_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(raw_path))[0]
    rgb_path = os.path.join(rgb_dir, base + '.png')
    if not os.path.isfile(rgb_path):
        srgb = demosaic_to_srgb_u16(raw_path)
        _atomic_imwrite(rgb_path, cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR))

    packed_path = None
    if is_short and packed_dir is not None:
        os.makedirs(packed_dir, exist_ok=True)
        packed_path = os.path.join(packed_dir, base + '.npy')
        if not os.path.isfile(packed_path):
            _atomic_save_npy(packed_path, pack_raw_file(raw_path, black_level, white_level).astype(np.float32))
    return rgb_path, packed_path


# =============================================================================
# Model (must match training.py exactly)
# =============================================================================
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
    )


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int=None):
        super().__init__()
        F_int = F_int or max(F_l // 2, 1)
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, stride=2), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        alpha = self.upsample(psi)
        if alpha.shape[-2:] != x.shape[-2:]:
            alpha = F.interpolate(alpha, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * alpha


class AttentionUNetGenerator(nn.Module, PyTorchModelHubMixin):
    def __init__(self, in_channels=4, enc_channels=(32, 64, 128, 256), bottleneck_channels=512):
        super().__init__()
        c1, c2, c3, c4 = enc_channels
        self.enc1 = conv_block(in_channels, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.bottleneck = conv_block(c4, bottleneck_channels)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att4 = AttentionGate(bottleneck_channels, c4)
        self.dec4 = conv_block(bottleneck_channels + c4, c4)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att3 = AttentionGate(c4, c3)
        self.dec3 = conv_block(c4 + c3, c3)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att2 = AttentionGate(c3, c2)
        self.dec2 = conv_block(c3 + c2, c2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att1 = AttentionGate(c2, c1)
        self.dec1 = conv_block(c2 + c1, c1)

        self.out_conv = nn.Conv2d(c1, 12, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        _, _, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        a4 = self.att4(b, e4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))

        d3 = self.up3(d4)
        a3 = self.att3(d4, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(d3, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(d2, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        out = self.pixel_shuffle(self.out_conv(d1))
        if pad_h or pad_w:
            out = out[:, :, :H * 2, :W * 2]
        return out


def load_generator(model_path, mc, device):
    """Load a generator from any of the formats this project can produce:
    an HF-style save_pretrained() folder, a bare state-dict file/folder, or
    a full training checkpoint dict (from training.py's save_checkpoint)."""
    G = AttentionUNetGenerator(in_channels=mc['in_channels'],
                                enc_channels=tuple(mc['enc_channels']),
                                bottleneck_channels=mc['bottleneck_channels'])
    if os.path.isdir(model_path):
        if _HF_AVAILABLE and os.path.isfile(os.path.join(model_path, 'config.json')):
            G = AttentionUNetGenerator.from_pretrained(model_path)
        else:
            state_path = None
            for candidate in ('pytorch_model.bin', 'model.pt', 'model.safetensors'):
                cp = os.path.join(model_path, candidate)
                if os.path.isfile(cp):
                    state_path = cp
                    break
            if state_path is None:
                raise FileNotFoundError(f"No recognized weight file found in {model_path}")
            if state_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state = load_file(state_path)
            else:
                state = torch.load(state_path, map_location=device)
            G.load_state_dict(state)
    else:
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt['G'] if isinstance(ckpt, dict) and 'G' in ckpt else ckpt
        G.load_state_dict(state)
    return G.to(device).eval()


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(fake_rgb01, gt_rgb01, lpips_model=None, device='cpu'):
    """fake_rgb01 / gt_rgb01: HxWx3 float32 arrays in [0,1]."""
    psnr = sk_psnr(gt_rgb01, fake_rgb01, data_range=1.0)
    ssim = sk_ssim(gt_rgb01, fake_rgb01, data_range=1.0, channel_axis=2)
    out = {'psnr': psnr, 'ssim': ssim}
    if lpips_model is not None:
        f = torch.from_numpy(fake_rgb01.transpose(2, 0, 1)).unsqueeze(0).float().to(device) * 2 - 1
        g = torch.from_numpy(gt_rgb01.transpose(2, 0, 1)).unsqueeze(0).float().to(device) * 2 - 1
        with torch.no_grad():
            out['lpips'] = lpips_model(f, g).item()
    return out


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Run the Attention U-Net GAN generator on SID Sony RAW input.")
    p.add_argument('--model_path', required=True,
                    help="HF save_pretrained() folder, a state-dict file, or a training.py checkpoint (.pt)")
    p.add_argument('--config', default='config.yaml', help="used for model architecture + default data paths")
    p.add_argument('--output_dir', required=True, help="where generated PNGs are saved")

    # batch mode
    p.add_argument('--data_root', default=None, help="override config.yaml data_root")
    p.add_argument('--list_file', default=None,
                    help="split list to iterate (default: config.yaml's test_list); "
                         "relative to data_root unless an absolute path is given")

    # single-file mode
    p.add_argument('--input', default=None, help="one-off .ARW short-exposure file (skips list iteration)")
    p.add_argument('--gt', default=None, help="matching long-exposure .ARW ground truth (single-file mode, for --eval)")
    p.add_argument('--ratio', type=float, default=None,
                    help="amplification ratio for single-file mode when --gt isn't given")

    eval_group = p.add_mutually_exclusive_group()
    eval_group.add_argument('--eval', dest='eval', action='store_true', default=True,
                             help="compute PSNR/SSIM/LPIPS against GT when available (default: on)")
    eval_group.add_argument('--no-eval', dest='eval', action='store_false',
                             help="turn off metrics computation")

    p.add_argument('--save_comparison', action='store_true',
                    help="also save a condition|generated|GT side-by-side PNG per sample")
    p.add_argument('--device', default=None)
    return p.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _to_uint8_bgr(rgb01):
    arr = np.clip(rgb01, 0, 1)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def run_one(short_path, long_path, ratio_override, G, cfg, device, out_dir,
            do_eval, lpips_model, save_comparison):
    d = cfg['data']
    bl, wl = d.get('black_level', 512), d.get('white_level', 16383)
    ratio_cap = d.get('ratio_cap', 300.0)
    data_root = d['data_root']
    short_rgb_dir = os.path.join(data_root, d['short_rgb_dirname'])
    long_rgb_dir = os.path.join(data_root, d['long_rgb_dirname'])
    packed_dir = os.path.join(data_root, d['packed_cache_dirname'])

    _, packed_path = ensure_rgb_cache(short_path, short_rgb_dir, bl, wl, is_short=True, packed_dir=packed_dir)
    packed = np.load(packed_path).astype(np.float32)

    if long_path is not None:
        ratio = compute_ratio(short_path, long_path, ratio_cap)
    elif ratio_override is not None:
        ratio = ratio_override
    else:
        raise ValueError(f"No GT and no --ratio given for {short_path}; can't determine amplification.")

    x = np.minimum(packed * ratio, 1.0)
    x_t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        fake = G(x_t).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)  # HWC RGB float32 [0,1]

    out_name = "test_" + os.path.splitext(os.path.basename(short_path))[0] + ".png"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, _to_uint8_bgr(fake))

    metrics = None
    if do_eval and long_path is not None:
        long_rgb_path, _ = ensure_rgb_cache(long_path, long_rgb_dir, bl, wl, is_short=False)
        gt_bgr = cv2.imread(long_rgb_path, cv2.IMREAD_UNCHANGED)
        gt_rgb01 = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        metrics = compute_metrics(fake, gt_rgb01, lpips_model, device)

    if save_comparison:
        cond_rgb_path, _ = ensure_rgb_cache(short_path, short_rgb_dir, bl, wl, is_short=False)
        cond_bgr = cv2.imread(cond_rgb_path, cv2.IMREAD_UNCHANGED)
        cond_rgb01 = cv2.cvtColor(cond_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        panels = [_to_uint8_bgr(cond_rgb01), _to_uint8_bgr(fake)]
        if long_path is not None:
            long_rgb_path, _ = ensure_rgb_cache(long_path, long_rgb_dir, bl, wl, is_short=False)
            gt_bgr = cv2.imread(long_rgb_path, cv2.IMREAD_UNCHANGED)
            gt_rgb01 = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
            panels.append(_to_uint8_bgr(gt_rgb01))
        h = min(p.shape[0] for p in panels)
        w = min(p.shape[1] for p in panels)
        panels = [cv2.resize(p, (w, h)) for p in panels]
        cmp_path = os.path.join(out_dir, "compare_" + os.path.splitext(os.path.basename(short_path))[0] + ".png")
        cv2.imwrite(cmp_path, np.concatenate(panels, axis=1))

    return out_path, metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg['data']['data_root'] = args.data_root
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] using {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    G = load_generator(args.model_path, cfg['model'], device)
    print(f"[model] loaded generator from {args.model_path}")

    lpips_model = None
    if args.eval:
        lpips_model = get_lpips_model(device)

    results = []

    if args.input is not None:
        # --- single-file mode ---
        out_path, metrics = run_one(args.input, args.gt, args.ratio, G, cfg, device,
                                     args.output_dir, args.eval, lpips_model, args.save_comparison)
        print(f"[saved] {out_path}")
        if metrics:
            print(f"[metrics] PSNR {metrics['psnr']:.3f} dB | SSIM {metrics['ssim']:.4f}"
                  + (f" | LPIPS {metrics['lpips']:.4f}" if 'lpips' in metrics else ""))
        return

    # --- batch mode over a split list ---
    data_root = cfg['data']['data_root']
    list_file = args.list_file or cfg['data'].get('test_list')
    if list_file is None:
        raise ValueError("No --list_file given and config.yaml has no data.test_list set.")
    list_path = list_file if os.path.isabs(list_file) else os.path.join(data_root, list_file)
    pairs = read_list_file(list_path)

    if cfg['data'].get('exclude_bad_pairs', True):
        bad = set(cfg['data'].get('exclude_ids', []))
        pairs = [p for p in pairs if parse_scene_id(p[0]) not in bad]

    print(f"[data] running inference on {len(pairs)} pairs from {list_path}")
    for i, (short_base, long_base) in enumerate(pairs):
        short_path = os.path.join(data_root, cfg['data']['short_dirname'], short_base)
        long_path = os.path.join(data_root, cfg['data']['long_dirname'], long_base)
        out_path, metrics = run_one(short_path, long_path, None, G, cfg, device,
                                     args.output_dir, args.eval, lpips_model, args.save_comparison)
        if metrics:
            results.append(metrics)
        if (i + 1) % 25 == 0 or (i + 1) == len(pairs):
            print(f"  [{i + 1}/{len(pairs)}] -> {os.path.basename(out_path)}")

    if args.eval and results:
        avg_psnr = float(np.mean([r['psnr'] for r in results]))
        avg_ssim = float(np.mean([r['ssim'] for r in results]))
        print("\n=== Test-set metrics ===")
        print(f"{'PSNR (dB)':>12} | {'SSIM':>8}" + (f" | {'LPIPS':>8}" if lpips_model is not None else ""))
        line = f"{avg_psnr:12.3f} | {avg_ssim:8.4f}"
        if lpips_model is not None:
            avg_lpips = float(np.mean([r['lpips'] for r in results]))
            line += f" | {avg_lpips:8.4f}"
        print(line)


if __name__ == "__main__":
    main()