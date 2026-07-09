#!/usr/bin/env python3
"""
validate.py

Runs validation on a single checkpoint.

Responsibilities:
- Load checkpoint
- Compute PSNR/SSIM
- Append results to validation_results.txt
- Update models/best.pt if SSIM improves
- Delete checkpoint afterwards

Usage:

python validate.py \
    --config config.yaml \
    --checkpoint output/models/checkpoints/epoch_0012.pt
"""

import argparse
import os
import shutil
import torch
from torch.utils.data import DataLoader

from training import (
    load_config,
    SIDDataset,
    AttentionUNetGenerator,
    load_checkpoint,
    run_validation,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    return parser.parse_args()


def append_result(results_path, epoch, psnr, ssim):

    first_time = not os.path.exists(results_path)

    with open(results_path, "a") as f:

        if first_time:

            f.write("=" * 55 + "\n")
            f.write("Validation Results\n")
            f.write("=" * 55 + "\n\n")

            f.write(
                f"{'Epoch':<10}"
                f"{'PSNR(dB)':<15}"
                f"{'SSIM':<12}\n"
            )

            f.write(
                f"{'-'*5:<10}"
                f"{'-'*8:<15}"
                f"{'-'*4:<12}\n"
            )

        f.write(
            f"{epoch:<10}"
            f"{psnr:<15.4f}"
            f"{ssim:<12.6f}\n"
        )


def load_best_metric(metric_file):

    if not os.path.exists(metric_file):
        return -1.0

    try:
        with open(metric_file, "r") as f:
            return float(f.read().strip())
    except Exception:
        return -1.0


def save_best_metric(metric_file, value):

    with open(metric_file, "w") as f:
        f.write(f"{value:.8f}")


def main():

    args = parse_args()

    cfg = load_config(args.config)

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ############################################################
    # Dataset
    ############################################################

    val_ds = SIDDataset(cfg, split="val")

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ############################################################
    # Model
    ############################################################

    mc = cfg["model"]

    G = AttentionUNetGenerator(
        in_channels=mc["in_channels"],
        enc_channels=tuple(mc["enc_channels"]),
        bottleneck_channels=mc["bottleneck_channels"],
    ).to(device)

    ############################################################
    # Load checkpoint
    ############################################################

    epoch, _ = load_checkpoint(
        args.checkpoint,
        G,
        D=None,
        map_location=device,
    )

    ############################################################
    # Validation
    ############################################################

    psnr, ssim = run_validation(
        G,
        val_loader,
        device,
    )

    ############################################################
    # Paths
    ############################################################

    output_dir = cfg["train"]["output_dir"]

    models_dir = os.path.join(output_dir, "models")

    os.makedirs(models_dir, exist_ok=True)

    results_file = os.path.join(
        output_dir,
        "validation_results.txt",
    )

    metric_file = os.path.join(
        models_dir,
        "best_metric.txt",
    )

    best_ckpt = os.path.join(
        models_dir,
        "best.pt",
    )

    ############################################################
    # Log metrics
    ############################################################

    append_result(
        results_file,
        epoch,
        psnr,
        ssim,
    )

    ############################################################
    # Check for best model
    ############################################################

    best_ssim = load_best_metric(metric_file)

    if ssim > best_ssim:

        shutil.copy2(
            args.checkpoint,
            best_ckpt,
        )

        save_best_metric(
            metric_file,
            ssim,
        )

        print(
            f"[validation] New best model! "
            f"SSIM={ssim:.6f}"
        )

    else:

        print(
            f"[validation] SSIM={ssim:.6f} "
            f"(best={best_ssim:.6f})"
        )

    ############################################################
    # Delete processed checkpoint
    ############################################################

    try:
        os.remove(args.checkpoint)
    except OSError:
        pass

    print(
        f"[validation] Finished epoch {epoch}"
    )


if __name__ == "__main__":
    main()