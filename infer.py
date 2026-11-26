#!/usr/bin/env python3
"""Restore paired noisy RGB and NIR images with a trained FreqNIR model."""

import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from freqnir import FreqNIR
from freqnir.data import discover_images, load_image
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
    parser.add_argument("--rgb", required=True, help="Noisy RGB image or directory")
    parser.add_argument("--nir", required=True, help="Aligned NIR image or directory")
    parser.add_argument("--output", default="results")
    parser.add_argument("--tile-size", type=int, default=256, help="0 enables full-frame attention")
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def image_map(path: Path) -> Dict[str, Path]:
    if path.is_file():
        return {path.stem: path}
    return discover_images(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FreqNIR(**model_kwargs(config)).to(device)
    load_checkpoint(model, args.checkpoint, map_location=str(device), strict=True)
    model.eval()

    rgb_files = image_map(Path(args.rgb))
    nir_files = image_map(Path(args.nir))
    keys = sorted(set(rgb_files) & set(nir_files))
    if not keys:
        raise RuntimeError("No RGB/NIR pairs with matching relative stems were found")

    output_dir = Path(args.output)
    with torch.no_grad():
        for key in tqdm(keys, desc="Inference"):
            rgb = load_image(rgb_files[key], 3).unsqueeze(0).to(device)
            nir = load_image(nir_files[key], 1).unsqueeze(0).to(device)
            if rgb.shape[-2:] != nir.shape[-2:]:
                raise ValueError(f"Spatial mismatch for '{key}'")
            prediction = tiled_forward(model, rgb, nir, args.tile_size, args.overlap)
            save_tensor_image(prediction[0], output_dir / f"{key}.png")


if __name__ == "__main__":
    main()
