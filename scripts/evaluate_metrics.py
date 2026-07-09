#!/usr/bin/env python3
"""
evaluate_metrics.py — offline PSNR / SSIM / LPIPS comparison between
generated (enhanced) images and SID Sony ground-truth long-exposure images.

Standalone by design, like inference.py: duplicates the small pieces it
needs (raw demosaic, scene-id parsing) so it can be run on its own against
any two folders of images. See evaluate_metrics_README.txt for the full
naming-scheme contract and usage examples.

Quick start:
    python evaluate_metrics.py --gen_dir ./results

By default this reads ground truth from config.yaml's data_root/long_dirname
(./Sony/long) and matches every file in --gen_dir to a ground-truth file by
shared 5-digit scene id (see --match_mode for the alternative exact-stem mode).

Optional extras via --extra_metrics (comma-separated): mae, dists, loe, fid.
See evaluate_metrics_README.txt for what each one means and what it needs.
"""
import os
import re
import csv
import glob
import argparse

import numpy as np
import cv2

try:
    import rawpy
except ImportError:
    rawpy = None

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import yaml
except ImportError:
    yaml = None

import torch

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None

try:
    import DISTS_pytorch
    _DISTS_AVAILABLE = True
except ImportError:
    _DISTS_AVAILABLE = False


def get_model_cache_dir():
    model_dir = os.environ.get('MODEL_DIR')
    if not model_dir:
        return None
    cache_dir = os.path.join(model_dir, 'lpips')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_lpips_model(device='cpu', cache_dir=None):
    if lpips_lib is None:
        raise RuntimeError("The 'lpips' package is required for this script but is not installed.")

    if cache_dir is None:
        cache_dir = get_model_cache_dir()

    if cache_dir is not None:
        print(f"[eval] using LPIPS cache dir: {cache_dir}")
        return lpips_lib.LPIPS(net='alex', model_path=cache_dir).to(device)
    return lpips_lib.LPIPS(net='alex').to(device)


def get_dists_model(device='cpu', cache_dir=None):
    if not _DISTS_AVAILABLE:
        return None

    if cache_dir is None:
        cache_dir = get_model_cache_dir()

    if cache_dir is not None:
        print(f"[eval] using DISTS cache dir: {cache_dir}")
        return DISTS_pytorch.DISTS(model_path=cache_dir).to(device)
    return DISTS_pytorch.DISTS().to(device)

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    _FID_AVAILABLE = True
except ImportError:
    _FID_AVAILABLE = False

import tempfile
import shutil


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
RAW_EXTS = ('.arw', '.cr2', '.nef', '.dng', '.raf')


# =============================================================================
# Naming / matching
# =============================================================================
def extract_scene_id(filename):
    """First run of 4-6 digits in the basename — the SID scene id, e.g.
    'test_00001_00_0.04s.png' -> 1, '00001_00_10s.ARW' -> 1."""
    m = re.search(r'(\d{4,6})', os.path.basename(filename))
    return int(m.group(1)) if m else None


def stem(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def build_matches(gt_files, gen_files, mode):
    """Returns [(gt_path, [gen_path, ...]), ...] — each GT can match several
    generated files (one per short-exposure burst) in 'scene' mode, or at
    most one in 'stem' mode."""
    if mode == 'stem':
        gen_by_key = {stem(g): g for g in gen_files}
        matches = []
        for gt in gt_files:
            g = gen_by_key.get(stem(gt))
            matches.append((gt, [g] if g else []))
        return matches

    gen_by_scene = {}
    for g in gen_files:
        sid = extract_scene_id(g)
        if sid is not None:
            gen_by_scene.setdefault(sid, []).append(g)

    matches = []
    for gt in gt_files:
        sid = extract_scene_id(gt)
        matches.append((gt, gen_by_scene.get(sid, [])))
    return matches


GEN_NAME_PREFIXES = ('test_', 'compare_')  # inference.py's output naming conventions


def find_input_match(gen_path, input_by_stem, input_by_scene):
    """Find the short-exposure input file a generated image was produced
    from, for LOE. Tries an exact stem match after stripping inference.py's
    known output prefixes (e.g. 'test_00001_00_0.04s.png' -> '00001_00_0.04s'),
    falling back to (ambiguous, first-match) scene-id matching otherwise."""
    s = stem(gen_path)
    for prefix in GEN_NAME_PREFIXES:
        if s.startswith(prefix) and s[len(prefix):] in input_by_stem:
            return input_by_stem[s[len(prefix):]]

    sid = extract_scene_id(gen_path)
    candidates = input_by_scene.get(sid, [])
    if len(candidates) > 1:
        print(f"  [warn] LOE: {len(candidates)} short-exposure candidates for scene {sid}, "
              f"using {os.path.basename(candidates[0])} (ambiguous match)")
    return candidates[0] if candidates else None


# =============================================================================
# Image loading (mirrors training.py / inference.py normalization)
# =============================================================================
def demosaic_to_srgb_u16(raw_path):
    if rawpy is None:
        raise RuntimeError("rawpy is required to read RAW ground-truth files but is not installed.")
    with rawpy.imread(raw_path) as raw:
        im = raw.postprocess(use_camera_wb=True, half_size=False,
                              no_auto_bright=True, output_bps=16)
    return im  # (H, W, 3) uint16, RGB order


def load_rgb01(path):
    """Load any supported RAW or standard image file as HxWx3 float32 RGB in [0,1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        img = demosaic_to_srgb_u16(path)  # uint16, RGB
        return img.astype(np.float32) / 65535.0
    img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    max_val = 65535.0 if img_rgb.dtype == np.uint16 else 255.0
    return img_rgb.astype(np.float32) / max_val


# =============================================================================
# Metrics
# =============================================================================
def compute_loe(input01, gen01, size=50):
    """Lightness Order Error (Guo et al., LIME, 2017). Measures whether the
    enhanced image preserves the input's pairwise relative-lightness order —
    NOT compared against ground truth, but against the original degraded
    input, per the metric's standard definition. Downsampled to `size`x`size`
    (the paper's own convention) since the pairwise comparison is O(N^2)."""
    if input01.shape[:2] != gen01.shape[:2]:
        gen01 = cv2.resize(gen01, (input01.shape[1], input01.shape[0]), interpolation=cv2.INTER_CUBIC)
        gen01 = np.clip(gen01, 0.0, 1.0)

    L_in = cv2.resize(np.max(input01, axis=2), (size, size), interpolation=cv2.INTER_AREA).reshape(-1)
    L_gen = cv2.resize(np.max(gen01, axis=2), (size, size), interpolation=cv2.INTER_AREA).reshape(-1)

    order_in = L_in[:, None] >= L_in[None, :]
    order_gen = L_gen[:, None] >= L_gen[None, :]
    return float(np.logical_xor(order_in, order_gen).sum() / (size * size))


def compute_fid(gt01_by_path, gen_paths, device):
    """Frechet Inception Distance, computed once over the whole matched set
    (a distributional metric — unlike PSNR/SSIM/LPIPS/LOE, it is not a
    per-pair score, so it isn't added to the per-pair rows/CSV)."""
    if not _FID_AVAILABLE:
        print("[warn] --extra_metrics includes 'fid' but the pytorch-fid package isn't installed "
              "— skipping FID (pip install pytorch-fid to enable).")
        return None

    real_dir = tempfile.mkdtemp(prefix='evalmetrics_real_')
    fake_dir = tempfile.mkdtemp(prefix='evalmetrics_fake_')
    try:
        for i, (gt_path, gt01) in enumerate(gt01_by_path.items()):
            arr = (np.clip(gt01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            cv2.imwrite(os.path.join(real_dir, f'{i:06d}.png'), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        for i, gen_path in enumerate(gen_paths):
            shutil.copy(gen_path, os.path.join(fake_dir, f'{i:06d}{os.path.splitext(gen_path)[1]}'))

        batch_size = min(50, len(gt01_by_path), len(gen_paths))
        return calculate_fid_given_paths([real_dir, fake_dir], batch_size, str(device), dims=2048)
    except Exception as e:
        print(f"[warn] FID computation failed ({e}) — skipping.")
        return None
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)


def compute_all_metrics(gt01, gen01, lpips_model, dists_model, device, extra, input01=None, loe_size=50):
    if gt01.shape[:2] != gen01.shape[:2]:
        print(f"  [warn] size mismatch GT {gt01.shape[:2]} vs generated {gen01.shape[:2]} "
              f"— resizing generated to GT size")
        gen01 = cv2.resize(gen01, (gt01.shape[1], gt01.shape[0]), interpolation=cv2.INTER_CUBIC)
        gen01 = np.clip(gen01, 0.0, 1.0)

    out = {
        'psnr': sk_psnr(gt01, gen01, data_range=1.0),
        'ssim': sk_ssim(gt01, gen01, data_range=1.0, channel_axis=2),
    }

    gt_t = torch.from_numpy(gt01.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    gen_t = torch.from_numpy(gen01.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out['lpips'] = lpips_model(gen_t * 2 - 1, gt_t * 2 - 1).item()
        if dists_model is not None:
            out['dists'] = dists_model(gen_t, gt_t).item()
    if 'mae' in extra:
        out['mae'] = float(np.mean(np.abs(gt01 - gen01)))
    if 'loe' in extra and input01 is not None:
        out['loe'] = compute_loe(input01, gen01, size=loe_size)
    return out


# =============================================================================
# CLI
# =============================================================================
def load_config(path):
    if yaml is None or not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Compare generated images against SID Sony ground truth (PSNR/SSIM/LPIPS).")
    p.add_argument('--gen_dir', required=True, help="folder of generated/enhanced images to evaluate")
    p.add_argument('--gt_dir', default=None,
                    help="folder of ground-truth files (RAW .ARW or already-demosaiced images); "
                         "default: config.yaml's data_root/long_dirname (./Sony/long)")
    p.add_argument('--config', default='config.yaml', help="only used to resolve the default --gt_dir")
    p.add_argument('--match_mode', choices=['scene', 'stem'], default='scene',
                    help="'scene' (default): match by shared 5-digit scene id, one GT to many generated "
                         "files. 'stem': exact basename match (excluding extension), one GT to one generated file.")
    p.add_argument('--lpips_net', default='alex', choices=['alex', 'vgg', 'squeeze'])
    p.add_argument('--extra_metrics', default='', help="comma-separated extras to add: mae,dists,loe,fid")
    p.add_argument('--input_dir', default=None,
                    help="folder of short-exposure input files (RAW or images), used only for LOE; "
                         "default: config.yaml's data_root/short_dirname (./Sony/short)")
    p.add_argument('--loe_size', type=int, default=50,
                    help="downsample size for LOE's pairwise lightness-order comparison (default: 50, per the LIME paper)")
    p.add_argument('--output_csv', default=None, help="path to write a per-pair CSV report (default: <gen_dir>/metrics_report.csv)")
    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    extra = set(x.strip().lower() for x in args.extra_metrics.split(',') if x.strip())
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] using {device}")

    gt_dir = args.gt_dir
    if gt_dir is None:
        cfg = load_config(args.config)
        if cfg is not None:
            gt_dir = os.path.join(cfg['data']['data_root'], cfg['data']['long_dirname'])
        else:
            gt_dir = './Sony/long'
    print(f"[data] ground truth dir: {gt_dir}")
    print(f"[data] generated dir:    {args.gen_dir}")

    gt_files = sorted(f for f in glob.glob(os.path.join(gt_dir, '*'))
                       if os.path.splitext(f)[1].lower() in RAW_EXTS + IMAGE_EXTS)
    gen_files = sorted(f for f in glob.glob(os.path.join(args.gen_dir, '*'))
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth files found in {gt_dir}")
    if not gen_files:
        raise FileNotFoundError(f"No generated images found in {args.gen_dir}")
    print(f"[data] {len(gt_files)} ground-truth files, {len(gen_files)} generated images, "
          f"match_mode={args.match_mode}")

    input_by_stem, input_by_scene = {}, {}
    if 'loe' in extra:
        input_dir = args.input_dir
        if input_dir is None:
            cfg = load_config(args.config)
            input_dir = os.path.join(cfg['data']['data_root'], cfg['data']['short_dirname']) if cfg else './Sony/short'
        input_files = [f for f in glob.glob(os.path.join(input_dir, '*'))
                       if os.path.splitext(f)[1].lower() in RAW_EXTS + IMAGE_EXTS]
        if not input_files:
            print(f"[warn] --extra_metrics includes 'loe' but no input files found in {input_dir} — skipping LOE.")
            extra.discard('loe')
        else:
            print(f"[data] LOE input dir: {input_dir} ({len(input_files)} files)")
            for f in input_files:
                input_by_stem[stem(f)] = f
                sid = extract_scene_id(f)
                if sid is not None:
                    input_by_scene.setdefault(sid, []).append(f)

    cache_dir = get_model_cache_dir()
    lpips_model = get_lpips_model(device=device, cache_dir=cache_dir)

    dists_model = None
    if 'dists' in extra:
        if not _DISTS_AVAILABLE:
            print("[warn] --extra_metrics includes 'dists' but the DISTS_pytorch package isn't installed "
                  "— skipping DISTS (pip install DISTS-pytorch to enable).")
        else:
            dists_model = get_dists_model(device=device, cache_dir=cache_dir)

    matches = build_matches(gt_files, gen_files, args.match_mode)

    rows = []
    n_unmatched_gt = 0
    matched_gen = set()
    gt01_cache = {}  # reused for FID below, so GTs aren't re-demosaiced
    for gt_path, gen_paths in matches:
        if not gen_paths:
            n_unmatched_gt += 1
            continue
        gt01 = load_rgb01(gt_path)
        gt01_cache[gt_path] = gt01
        for gen_path in gen_paths:
            matched_gen.add(gen_path)
            gen01 = load_rgb01(gen_path)

            input01 = None
            if 'loe' in extra:
                input_path = find_input_match(gen_path, input_by_stem, input_by_scene)
                if input_path is not None:
                    input01 = load_rgb01(input_path)
                else:
                    print(f"  [warn] LOE: no matching input file for {os.path.basename(gen_path)} — skipping LOE for this pair.")

            m = compute_all_metrics(gt01, gen01, lpips_model, dists_model, device, extra,
                                     input01=input01, loe_size=args.loe_size)
            rows.append({'gt': os.path.basename(gt_path), 'generated': os.path.basename(gen_path), **m})
            extras_str = ''.join(f" | {k.upper()} {m[k]:.4f}" for k in m if k not in ('psnr', 'ssim', 'lpips'))
            print(f"  {os.path.basename(gen_path):40s} vs {os.path.basename(gt_path):20s} "
                  f"PSNR {m['psnr']:6.3f} dB | SSIM {m['ssim']:.4f} | LPIPS {m['lpips']:.4f}{extras_str}")

    n_unmatched_gen = len(gen_files) - len(matched_gen)
    if n_unmatched_gt:
        print(f"[warn] {n_unmatched_gt} ground-truth file(s) had no matching generated image — skipped.")
    if n_unmatched_gen:
        print(f"[warn] {n_unmatched_gen} generated file(s) had no matching ground truth — skipped.")

    if not rows:
        print("[done] no pairs were evaluated.")
        return

    keys = []  # union across all rows, preserving first-seen order — 'loe' may be
    for r in rows:  # absent on some rows (no matching input file found for LOE)
        for k in r:
            if k not in ('gt', 'generated') and k not in keys:
                keys.append(k)
    averages = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}

    print("\n=== Summary (mean +/- std over {} pairs) ===".format(len(rows)))
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        n_note = f" (n={len(vals)})" if len(vals) != len(rows) else ""
        print(f"  {k.upper():6s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{n_note}")

    print("\n=== Average (paper metrics) ===")
    print(f"{'PSNR (dB)':>12} | {'SSIM':>8} | {'LPIPS':>8}")
    print(f"{averages['psnr']:12.3f} | {averages['ssim']:8.4f} | {averages['lpips']:8.4f}")

    if 'fid' in extra:
        fid_value = compute_fid(gt01_cache, sorted(matched_gen), device)
        if fid_value is not None:
            print("\n=== FID (set-level, not per-pair — computed over the {} matched images) ===".format(len(matched_gen)))
            print(f"  FID: {fid_value:.4f}")
            if len(matched_gen) < 2048:
                print(f"  [note] FID's covariance estimate is unreliable with few samples "
                      f"(you have {len(matched_gen)}; the literature typically uses 2000+). "
                      f"Treat this number as indicative, not a paper-comparable score.")

    out_csv = args.output_csv or os.path.join(args.gen_dir, 'metrics_report.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['gt', 'generated'] + keys)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({'gt': '', 'generated': 'AVERAGE', **averages})
    print(f"[saved] per-pair report (+ AVERAGE row) -> {out_csv}")


if __name__ == "__main__":
    main()
