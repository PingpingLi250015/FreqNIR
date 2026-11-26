#!/usr/bin/env python3
"""Evaluate a trained FreqNIR checkpoint with PSNR and SSIM."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freqnir import FreqNIR
from freqnir.data import PairedRGBNIRDataset
from freqnir.metrics import image_metrics
from freqnir.utils import (
    load_checkpoint,
    load_config,
    model_kwargs,
    save_tensor_image,
    tiled_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Folder containing noisy_rgb/nir/gt")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--save-images", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FreqNIR(**model_kwargs(config)).to(device)
    load_checkpoint(model, args.checkpoint, map_location=str(device), strict=True)
    model.eval()
    loader = DataLoader(PairedRGBNIRDataset(args.data), batch_size=1, shuffle=False)

    totals = {"psnr": 0.0, "ssim": 0.0}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            target = batch["target"].to(device)
            prediction = tiled_forward(model, rgb, nir, args.tile_size, args.overlap)
            scores = image_metrics(prediction, target)
            totals["psnr"] += scores["psnr"].item()
            totals["ssim"] += scores["ssim"].item()
            if args.save_images:
                save_tensor_image(prediction[0], Path(args.save_images) / f"{batch['name'][0]}.png")

    count = len(loader.dataset)
    print(f"PSNR: {totals['psnr'] / count:.4f} dB")
    print(f"SSIM: {totals['ssim'] / count:.6f}")


if __name__ == "__main__":
    main()
