#!/usr/bin/env python3
"""
training.py — Attention U-Net GAN for SID Sony low-light RAW enhancement.

Standalone by design: dataset, disk-cache preprocessing, model definitions
(Generator + Discriminator), losses, checkpoint I/O and the training loop
all live in this one file. The only other things this project needs are
`config.yaml` and `requirements.txt` (see the sibling `inference.py`, which
is likewise self-contained and duplicates the small pieces it needs so it
can run standalone against a trained model + a data folder).

Expected data_root layout (see config.yaml `data:` block to rename any of
these):

    <data_root>/
    ├── short/                  # input, short-exposure .ARW  (required)
    ├── long/                   # ground truth, long-exposure .ARW (required)
    ├── short_rgb/              # auto-built cache: sRGB demosaic of short/ (D condition)
    ├── long_rgb/                # auto-built cache: sRGB demosaic of long/  (GT / training target)
    ├── short_packed_cache/      # auto-built cache: packed 4ch raw (.npy), unamplified
    ├── Sony_train_list.txt      # optional official SID split files
    ├── Sony_val_list.txt
    └── Sony_test_list.txt

Usage:
    python training.py --config config.yaml
    python training.py --config config.yaml --resume_path ./models/latest.pt
    python training.py --config config.yaml --prepare_cache_only   # just build the disk cache and exit
"""
import os
import glob
import time
import random
import argparse

import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import rawpy
except ImportError:
    rawpy = None

from pytorch_msssim import MS_SSIM
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from huggingface_hub import PyTorchModelHubMixin
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

    class PyTorchModelHubMixin:  # no-op fallback so the class def below still works
        pass


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# RAW <-> tensor helpers (implementation-plan §2)
# =============================================================================
def pack_raw_array(raw_visible, black_level=512, white_level=16383):
    """RGGB Bayer -> 4-channel, half spatial resolution. `raw_visible` is the
    raw numpy array straight off the sensor (rawpy's `.raw_image_visible`)."""
    im = raw_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level - black_level)
    im = np.expand_dims(im, axis=2)
    H, W, _ = im.shape
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                           im[0:H:2, 1:W:2, :],
                           im[1:H:2, 1:W:2, :],
                           im[1:H:2, 0:W:2, :]), axis=2)
    return out  # (H/2, W/2, 4) float32 in [0,1], UNAMPLIFIED


def pack_raw_file(path, black_level=512, white_level=16383):
    if rawpy is None:
        raise RuntimeError("rawpy is required to read .ARW files but is not installed.")
    with rawpy.imread(path) as raw:
        arr = raw.raw_image_visible.copy()
    return pack_raw_array(arr, black_level, white_level)


def demosaic_to_srgb_u16(raw_path):
    """Full sRGB demosaic (camera WB, no amplification), 16-bit, full sensor
    resolution. Used to build both the D-conditioning image (short exposure)
    and the ground-truth target image (long exposure)."""
    if rawpy is None:
        raise RuntimeError("rawpy is required to build the RGB cache but is not installed.")
    with rawpy.imread(raw_path) as raw:
        im = raw.postprocess(use_camera_wb=True, half_size=False,
                              no_auto_bright=True, output_bps=16)
    return im  # (H, W, 3) uint16, RGB order


def parse_exposure(filename):
    """'00001_00_0.04s.ARW' -> 0.04"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    tok = stem.split('_')[-1]
    return float(tok[:-1]) if tok.endswith('s') else float(tok)


def parse_scene_id(filename):
    return int(os.path.basename(filename).split('_')[0])


def compute_ratio(short_path, long_path, cap=300.0):
    in_exp = parse_exposure(short_path)
    gt_exp = parse_exposure(long_path)
    return min(gt_exp / in_exp, cap)


# =============================================================================
# Split / pairing logic
# =============================================================================
def read_list_file(list_path):
    """Parse an SID split list file, e.g.
    './Sony/short/00001_00_0.04s.ARW ./Sony/long/00001_00_10s.ARW ISO200 F8'
    Returns [(short_basename, long_basename), ...] — resolved by basename so
    it works whether your data_root mirrors the nested `Sony/short/...` paths
    or is flat (data_root/short, data_root/long directly)."""
    pairs = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            short_base = os.path.basename(parts[0])
            long_base = os.path.basename(parts[1])
            pairs.append((short_base, long_base))
    return pairs


def scan_pairs_by_scene(short_dir, long_dir):
    """Fallback pairing (no list file available): match every short exposure
    to the long exposure sharing its 5-digit scene id, mirroring the original
    SID codebase's own glob-by-scene-id approach."""
    scene_to_long = {}
    for lf in sorted(glob.glob(os.path.join(long_dir, '*'))):
        scene_to_long.setdefault(parse_scene_id(lf), os.path.basename(lf))
    pairs = []
    for sf in sorted(glob.glob(os.path.join(short_dir, '*'))):
        sid = parse_scene_id(sf)
        if sid in scene_to_long:
            pairs.append((os.path.basename(sf), scene_to_long[sid]))
    return pairs


def get_split_pairs(cfg, split):
    d = cfg['data']
    data_root = d['data_root']
    list_name = d.get({'train': 'train_list', 'val': 'val_list', 'test': 'test_list'}[split])

    pairs = None
    if list_name:
        list_path = os.path.join(data_root, list_name)
        if os.path.isfile(list_path):
            pairs = read_list_file(list_path)

    if pairs is None:
        all_pairs = scan_pairs_by_scene(
            os.path.join(data_root, d['short_dirname']),
            os.path.join(data_root, d['long_dirname']))
        scenes = sorted({parse_scene_id(p[0]) for p in all_pairs})
        rnd = random.Random(cfg['train']['seed'])
        rnd.shuffle(scenes)
        n = len(scenes)
        n_val = max(1, int(0.1 * n))
        n_test = max(1, int(0.1 * n))
        val_scenes = set(scenes[:n_val])
        test_scenes = set(scenes[n_val:n_val + n_test])
        train_scenes = set(scenes[n_val + n_test:])
        target = {'train': train_scenes, 'val': val_scenes, 'test': test_scenes}[split]
        pairs = [p for p in all_pairs if parse_scene_id(p[0]) in target]

    if d.get('exclude_bad_pairs', True):
        bad = set(d.get('exclude_ids', []))
        pairs = [p for p in pairs if parse_scene_id(p[0]) not in bad]
    return pairs


# =============================================================================
# Disk cache (packed raw .npy + sRGB .png), built once and reused
# =============================================================================
def _atomic_save_npy(path, arr):
    tmp_path = path + f".tmp{os.getpid()}"
    np.save(tmp_path, arr)          # numpy appends .npy since tmp_path doesn't already end with it
    os.replace(tmp_path + ".npy", path)


def _atomic_imwrite(path, img):
    tmp_path = path + f".tmp{os.getpid()}.png"
    if not cv2.imwrite(tmp_path, img):
        raise RuntimeError(f"cv2.imwrite failed for {tmp_path}")
    os.replace(tmp_path, path)


def _packed_cache_is_valid(path):
    if not os.path.isfile(path):
        return False
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"[cache] invalid packed cache {path}: {e}")
        return False
    return isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] == 4


def _image_cache_is_valid(path):
    if not os.path.isfile(path):
        return False
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img is not None and img.size > 0


def ensure_cache_for_pair(short_base, long_base, cfg, build_missing=True):
    """Make sure the packed-.npy cache (short) and the sRGB .png caches
    (short + long) exist on disk for one pair, mirroring short/ and long/
    with matching basenames in their own labeled subfolders. Builds whatever
    is missing; a no-op if everything's already cached."""
    d = cfg['data']
    root = d['data_root']
    short_path = os.path.join(root, d['short_dirname'], short_base)
    long_path = os.path.join(root, d['long_dirname'], long_base)

    packed_dir = os.path.join(root, d['packed_cache_dirname'])
    short_rgb_dir = os.path.join(root, d['short_rgb_dirname'])
    long_rgb_dir = os.path.join(root, d['long_rgb_dirname'])
    os.makedirs(packed_dir, exist_ok=True)
    os.makedirs(short_rgb_dir, exist_ok=True)
    os.makedirs(long_rgb_dir, exist_ok=True)

    packed_path = os.path.join(packed_dir, os.path.splitext(short_base)[0] + '.npy')
    short_rgb_path = os.path.join(short_rgb_dir, os.path.splitext(short_base)[0] + '.png')
    long_rgb_path = os.path.join(long_rgb_dir, os.path.splitext(long_base)[0] + '.png')

    if build_missing:
        bl, wl = d.get('black_level', 512), d.get('white_level', 16383)
        if not _packed_cache_is_valid(packed_path):
            if os.path.isfile(packed_path):
                os.remove(packed_path)
            print(f"[cache] rebuilding packed cache {packed_path}")
            _atomic_save_npy(packed_path, pack_raw_file(short_path, bl, wl).astype(np.float32))
        if not _image_cache_is_valid(short_rgb_path):
            if os.path.isfile(short_rgb_path):
                os.remove(short_rgb_path)
            print(f"[cache] rebuilding short RGB cache {short_rgb_path}")
            srgb = demosaic_to_srgb_u16(short_path)
            _atomic_imwrite(short_rgb_path, cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR))
        if not _image_cache_is_valid(long_rgb_path):
            if os.path.isfile(long_rgb_path):
                os.remove(long_rgb_path)
            print(f"[cache] rebuilding long RGB cache {long_rgb_path}")
            lrgb = demosaic_to_srgb_u16(long_path)
            _atomic_imwrite(long_rgb_path, cv2.cvtColor(lrgb, cv2.COLOR_RGB2BGR))

    return dict(short_path=short_path, long_path=long_path, packed_path=packed_path,
                short_rgb_path=short_rgb_path, long_rgb_path=long_rgb_path)


def prepare_cache(pairs_by_split, cfg):
    """Single-process pre-pass that builds the entire disk cache up front.
    Doing this in the main process (before DataLoader workers spin up)
    avoids two workers racing to write the same file — some long-exposure
    GTs are shared by several short-exposure bursts."""
    seen, todo = set(), []
    for pairs in pairs_by_split.values():
        for p in pairs:
            if p not in seen:
                seen.add(p)
                todo.append(p)
    print(f"[cache] checking/building disk cache for {len(todo)} pairs ...")
    for i, (sb, lb) in enumerate(todo):
        ensure_cache_for_pair(sb, lb, cfg, build_missing=True)
        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            print(f"  [{i + 1}/{len(todo)}] cached")


class SIDDataset(Dataset):
    def __init__(self, cfg, split='train', use_ram_cache=True):
        self.cfg = cfg
        self.split = split
        self.pairs = get_split_pairs(cfg, split)
        self.ps = cfg['patch']['packed_patch_size']
        self.augment = cfg['patch'].get('augment', True) and split == 'train'
        self.ratio_cap = cfg['data'].get('ratio_cap', 300.0)
        
        # In-memory RAM cache to completely eliminate disk I/O after epoch 1
        self.use_ram_cache = use_ram_cache
        self.ram_cache = {}

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path, is_npy=False, max_attempts=4):
        if self.use_ram_cache and path in self.ram_cache:
            return self.ram_cache[path]

        # Retry on transient read glitches (seen intermittently on this shared
        # filesystem as libpng CRC errors that don't reproduce on re-read —
        # files pass full offline validation, so this isn't real corruption).
        last_err = None
        for attempt in range(max_attempts):
            try:
                if is_npy:
                    data = np.load(path, allow_pickle=False)
                else:
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is None or img.size == 0:
                        raise IOError(f"cv2.imread returned empty for {path}")
                    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.use_ram_cache:
                    self.ram_cache[path] = data
                return data
            except Exception as e:
                last_err = e
                print(f"[warn] transient read failure on {path} (attempt {attempt + 1}/{max_attempts}): {e}")
        raise RuntimeError(f"Failed to load {path} after {max_attempts} attempts") from last_err

    def __getitem__(self, idx):
        sb, lb = self.pairs[idx]
        d = self.cfg['data']
        root = d['data_root']

        # Construct paths directly to bypass the redundant disk validation
        packed_path = os.path.join(root, d['packed_cache_dirname'], os.path.splitext(sb)[0] + '.npy')
        short_rgb_path = os.path.join(root, d['short_rgb_dirname'], os.path.splitext(sb)[0] + '.png')
        long_rgb_path = os.path.join(root, d['long_rgb_dirname'], os.path.splitext(lb)[0] + '.png')
        short_path = os.path.join(root, d['short_dirname'], sb)
        long_path = os.path.join(root, d['long_dirname'], lb)

        # Load from RAM cache or disk
        packed = self._load_image(packed_path, is_npy=True)
        y_uint16 = self._load_image(long_rgb_path, is_npy=False)
        cond_uint16 = self._load_image(short_rgb_path, is_npy=False)

        ratio = compute_ratio(short_path, long_path, self.ratio_cap)

        if self.split == 'train':
            # CROP FIRST: Only process the pixels we actually need
            ps = min(self.ps, packed.shape[0], packed.shape[1])
            top = random.randint(0, packed.shape[0] - ps)
            left = random.randint(0, packed.shape[1] - ps)

            packed_crop = packed[top:top + ps, left:left + ps, :]
            y_crop = y_uint16[top * 2:top * 2 + ps * 2, left * 2:left * 2 + ps * 2, :]
            cond_crop = cond_uint16[top * 2:top * 2 + ps * 2, left * 2:left * 2 + ps * 2, :]

            # CONVERT LATER: Now convert just the tiny cropped patch to float32
            x = np.minimum(packed_crop * ratio, 1.0)
            y = y_crop.astype(np.float32) / 65535.0
            cond = cond_crop.astype(np.float32) / 65535.0

            if self.augment:
                if random.random() < 0.5:
                    x, y, cond = x[:, ::-1, :], y[:, ::-1, :], cond[:, ::-1, :]
                if random.random() < 0.5:
                    x, y, cond = x[::-1, :, :], y[::-1, :, :], cond[::-1, :, :]
                if random.random() < 0.5:
                    x, y, cond = x.transpose(1, 0, 2), y.transpose(1, 0, 2), cond.transpose(1, 0, 2)
        else:
            # Val logic remains the same (needs full image)
            x = np.minimum(packed * ratio, 1.0)
            y = y_uint16.astype(np.float32) / 65535.0
            cond = cond_uint16.astype(np.float32) / 65535.0

        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)))
        cond = torch.from_numpy(np.ascontiguousarray(cond.transpose(2, 0, 1)))
        
        return {'x': x, 'y': y, 'cond': cond, 'ratio': ratio,
                'short_path': short_path, 'long_path': long_path}


# =============================================================================
# Models — Attention U-Net Generator + PatchGAN Discriminator (plan §3)
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
        F_int = F_int or max(F_l // 2, 1)      # [ASSUMPTION] F_int = F_l // 2, per plan §3.1
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, stride=2), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)                        # downsample x to g's resolution
        psi = self.psi(self.relu(g1 + x1))
        alpha = self.upsample(psi)               # back to x's resolution
        if alpha.shape[-2:] != x.shape[-2:]:      # odd-size safety net
            alpha = F.interpolate(alpha, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * alpha                          # Eq. (2): x_hat_l = x_l * sigma_g(...)


class AttentionUNetGenerator(nn.Module, PyTorchModelHubMixin):
    """Attention U-Net generator. Input: packed 4-channel Bayer raw (already
    amplified + clipped). Output: 3-channel sRGB at full sensor resolution
    (2x the input's spatial size, undone via PixelShuffle)."""

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
        # Pad to a multiple of 16 (4 poolings) so full-resolution val/inference
        # images (which aren't cropped to a convenient size) don't break the
        # skip-connection concatenations; crop back to the exact size after.
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
        return out                                  # no final activation — clip at loss/inference time


class PatchDiscriminator(nn.Module):
    """70x70 Markovian PatchGAN (pix2pix-style). Conditioned on a 3-channel
    sRGB image; concatenated with the (real or fake) 3-channel sRGB image."""

    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()
        c = base_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c * 4, c * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, cond, img):
        return self.model(torch.cat([cond, img], dim=1))


# =============================================================================
# Losses (plan §4)
# =============================================================================
class GANLosses:
    def __init__(self, cfg, device):
        self.bce = nn.BCEWithLogitsLoss()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
        lc = cfg['loss']
        self.lambda_l1 = lc['lambda_l1']
        self.lambda_ms = lc['lambda_ms_ssim']
        self.lambda_total = lc['lambda_rec_total']
        self.lambda_adv = lc['lambda_adv']

    def d_loss(self, d_real_logits, d_fake_logits):
        return (self.bce(d_real_logits, torch.ones_like(d_real_logits)) +
                self.bce(d_fake_logits, torch.zeros_like(d_fake_logits)))

    def g_loss(self, d_fake_logits, fake, real):
        # NOTE: fake is used un-clamped here by design (plan §6.5d — clamping
        # is only for logging/visualization, not inside the loss).
        adv = self.bce(d_fake_logits, torch.ones_like(d_fake_logits))
        l1 = F.l1_loss(fake, real)
        ms = 1.0 - self.ms_ssim(fake, real)
        rec = self.lambda_l1 * l1 + self.lambda_ms * ms
        total = self.lambda_adv * adv + self.lambda_total * rec
        return total, {'adv': adv.item(), 'l1': l1.item(), 'ms_ssim': ms.item(), 'total': total.item()}


# =============================================================================
# Checkpoint I/O
# =============================================================================
def save_checkpoint(path, epoch, G, D, opt_G, opt_D, best_metric):
    torch.save({'epoch': epoch, 'G': G.state_dict(), 'D': D.state_dict(),
                'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
                'best_metric': best_metric}, path)


def load_checkpoint(path, G, D, opt_G=None, opt_D=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    G.load_state_dict(ckpt['G'])
    D.load_state_dict(ckpt['D'])
    if opt_G is not None and 'opt_G' in ckpt:
        opt_G.load_state_dict(ckpt['opt_G'])
    if opt_D is not None and 'opt_D' in ckpt:
        opt_D.load_state_dict(ckpt['opt_D'])
    return ckpt.get('epoch', 0), ckpt.get('best_metric', None)


def save_generator_hf(generator, out_dir):
    """Save the generator as an HF-style model folder (config.json +
    safetensors weights) when huggingface_hub is available; falls back to a
    plain state dict otherwise."""
    os.makedirs(out_dir, exist_ok=True)
    if _HF_AVAILABLE and hasattr(generator, 'save_pretrained'):
        generator.save_pretrained(out_dir)
    else:
        torch.save(generator.state_dict(), os.path.join(out_dir, 'pytorch_model.bin'))


# =============================================================================
# Visualization / validation helpers
# =============================================================================
def _tensor_to_uint8_bgr(t):
    arr = t.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_sample_triptych(cond, fake, real, out_dir, epoch, step, max_side=768):
    c = _tensor_to_uint8_bgr(cond[0])
    f = _tensor_to_uint8_bgr(fake[0])
    r = _tensor_to_uint8_bgr(real[0])
    h, w = r.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        c, f, r = cv2.resize(c, (nw, nh)), cv2.resize(f, (nw, nh)), cv2.resize(r, (nw, nh))
    triptych = np.concatenate([c, f, r], axis=1)   # condition | generated | GT
    cv2.imwrite(os.path.join(out_dir, f'epoch{epoch:04d}_step{step:07d}.png'), triptych)


def run_validation(G, val_loader, device):
    G.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            fake = G(x).clamp(0, 1)
            fake_np = fake[0].cpu().numpy().transpose(1, 2, 0)
            y_np = y[0].cpu().numpy().transpose(1, 2, 0)
            psnrs.append(sk_psnr(y_np, fake_np, data_range=1.0))
            ssims.append(sk_ssim(y_np, fake_np, data_range=1.0, channel_axis=2))
    G.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


# =============================================================================
# Config / CLI
# =============================================================================
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Train the Attention U-Net GAN on SID Sony.")
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--data_root', default=None)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--val_every_epochs', type=int, default=None,
                    help="override config.yaml's train.val_every_epochs (how often PSNR/SSIM validation runs)")
    p.add_argument('--skip_cache_check', action='store_true',
                    help="skip the startup prepare_cache() validation pass entirely — only safe once you've "
                         "already confirmed the disk cache is fully built and clean (e.g. a prior run completed it)")
    p.add_argument('--resume_path', default=None,
                    help="explicit checkpoint to resume from (default: <output_dir>/latest.pt if present)")
    p.add_argument('--no_resume', action='store_true', help="ignore any existing checkpoint, start fresh")
    p.add_argument('--prepare_cache_only', action='store_true', help="just build the disk cache, then exit")
    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg['data']['data_root'] = args.data_root
    if args.output_dir:
        cfg['train']['output_dir'] = args.output_dir
    if args.epochs:
        cfg['train']['epochs'] = args.epochs
    if args.batch_size:
        cfg['train']['batch_size'] = args.batch_size
    if args.val_every_epochs:
        cfg['train']['val_every_epochs'] = args.val_every_epochs
    if args.no_resume:
        cfg['train']['resume'] = False

    set_seed(cfg['train']['seed'])
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] using {device}")

    output_dir = cfg['train']['output_dir']
    models_dir = os.path.join(output_dir, 'models')
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, 'tb_logs')) if SummaryWriter is not None else None

    train_pairs = get_split_pairs(cfg, 'train')
    val_pairs = get_split_pairs(cfg, 'val')
    print(f"[data] train pairs: {len(train_pairs)} | val pairs: {len(val_pairs)}")
    if args.skip_cache_check:
        print("[cache] --skip_cache_check set, trusting existing disk cache without re-validating.")
    else:
        prepare_cache({'train': train_pairs, 'val': val_pairs}, cfg)

    if args.prepare_cache_only:
        print("[cache] --prepare_cache_only set, exiting after cache build.")
        return

    # use_ram_cache=False: the unbounded per-worker RAM cache in SIDDataset
    # (~163GB for the train split) doesn't fit this machine's 62GB system RAM
    # and, with num_workers>0 and no persistent_workers, gets discarded every
    # epoch anyway — disabled here rather than in training.py to avoid
    # touching the shared file.
    train_ds = SIDDataset(cfg, split='train', use_ram_cache=False)
    val_ds = SIDDataset(cfg, split='val', use_ram_cache=False)
    default_workers = max(1, os.cpu_count() // 2)
    num_workers = int(cfg['train'].get('num_workers', default_workers))
    print(f"[data] using {num_workers} DataLoader worker(s)")
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    mc = cfg['model']
    G = AttentionUNetGenerator(in_channels=mc['in_channels'],
                                enc_channels=tuple(mc['enc_channels']),
                                bottleneck_channels=mc['bottleneck_channels']).to(device)
    D = PatchDiscriminator(in_channels=mc['disc_in_channels'],
                            base_channels=mc['disc_base_channels']).to(device)

    oc = cfg['optim']
    opt_G = torch.optim.Adam(G.parameters(), lr=oc['lr_g'], betas=(oc['beta1'], oc['beta2']))
    opt_D = torch.optim.Adam(D.parameters(), lr=oc['lr_d'], betas=(oc['beta1'], oc['beta2']))
    losses = GANLosses(cfg, device)

    start_epoch, best_psnr = 1, -1.0
    latest_ckpt = args.resume_path or os.path.join(output_dir, 'latest.pt')
    if cfg['train']['resume'] and os.path.isfile(latest_ckpt):
        ep, best = load_checkpoint(latest_ckpt, G, D, opt_G, opt_D, map_location=device)
        start_epoch, best_psnr = ep + 1, (best if best is not None else -1.0)
        print(f"[resume] loaded {latest_ckpt}, continuing from epoch {start_epoch}")

    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        G.train(); D.train()
        t0 = time.time()
        for batch in train_loader:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            cond = batch['cond'].to(device, non_blocking=True)

            # --- Shared G Forward Pass ---
            # Run the generator once. 
            fake = G(x)

            # --- D step ---
            opt_D.zero_grad()
            d_real = D(cond, y)
            
            # Use .detach() here so gradients don't flow back into G during D's step
            d_fake = D(cond, fake.detach()) 
            
            loss_d = losses.d_loss(d_real, d_fake)
            loss_d.backward()
            opt_D.step()

            # --- G step ---
            opt_G.zero_grad()
            
            # Feed the exact same 'fake' (still attached to the graph) to D
            d_fake_for_g = D(cond, fake) 
            loss_g, parts = losses.g_loss(d_fake_for_g, fake, y)
            loss_g.backward()
            opt_G.step()

            global_step += 1
            if global_step % cfg['train']['log_every'] == 0:
                print(f"epoch {epoch} step {global_step} | D {loss_d.item():.4f} | "
                      f"G {parts['total']:.4f} (adv {parts['adv']:.4f} "
                      f"l1 {parts['l1']:.4f} ms {parts['ms_ssim']:.4f})")
                if writer:
                    writer.add_scalar('loss/D', loss_d.item(), global_step)
                    writer.add_scalar('loss/G_total', parts['total'], global_step)
                    writer.add_scalar('loss/G_adv', parts['adv'], global_step)
                    writer.add_scalar('loss/G_l1', parts['l1'], global_step)
                    writer.add_scalar('loss/G_ms_ssim', parts['ms_ssim'], global_step)

            if global_step % cfg['train']['sample_every'] == 0:
                save_sample_triptych(cond, fake, y, samples_dir, epoch, global_step)

        print(f"[epoch {epoch}] done in {time.time() - t0:.1f}s")

        if epoch % cfg['train']['val_every_epochs'] == 0 or epoch == cfg['train']['epochs']:
            val_psnr, val_ssim = run_validation(G, val_loader, device)
            print(f"[val @ epoch {epoch}] PSNR {val_psnr:.3f} dB | SSIM {val_ssim:.4f}")
            if writer:
                writer.add_scalar('val/psnr', val_psnr, epoch)
                writer.add_scalar('val/ssim', val_ssim, epoch)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_generator_hf(G, os.path.join(models_dir, 'best'))
                print(f"[val] new best PSNR {best_psnr:.3f} dB -> saved models/best")

        if epoch % cfg['train']['save_every_epochs'] == 0 or epoch == cfg['train']['epochs']:
            save_checkpoint(os.path.join(output_dir, 'latest.pt'), epoch, G, D, opt_G, opt_D, best_psnr)
            save_generator_hf(G, os.path.join(models_dir, f'epoch_{epoch:04d}'))
            print(f"[ckpt] saved latest.pt + models/epoch_{epoch:04d}")

    print("[done] training complete.")


if __name__ == "__main__":
    main()