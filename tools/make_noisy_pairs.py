#!/usr/bin/env python3
"""Create paired Gaussian or Poisson-Gaussian RGB data without downloading a dataset."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from freqnir.data import discover_images, load_image
from freqnir.utils import save_tensor_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-rgb", required=True)
    parser.add_argument("--nir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gaussian-sigma", type=float, default=4.0, help="8-bit sigma; values <=1 are normalized"
    )
    parser.add_argument("--poisson-alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    clean_files = discover_images(Path(args.clean_rgb))
    nir_files = discover_images(Path(args.nir))
    keys = sorted(set(clean_files) & set(nir_files))
    if not keys:
        raise RuntimeError("No clean-RGB/NIR pairs with matching relative stems were found")

    sigma = args.gaussian_sigma / 255.0 if args.gaussian_sigma > 1.0 else args.gaussian_sigma
    output = Path(args.output)
    for key in tqdm(keys, desc="Noise synthesis"):
        clean = load_image(clean_files[key], 3)
        nir = load_image(nir_files[key], 1)
        if clean.shape[-2:] != nir.shape[-2:]:
            raise ValueError(f"Spatial mismatch for '{key}'")
        standard_deviation = (sigma**2 + args.poisson_alpha * clean).sqrt()
        noise = torch.randn(clean.shape, generator=generator) * standard_deviation
        noisy = (clean + noise).clamp(0.0, 1.0)
        save_tensor_image(noisy, output / "noisy_rgb" / f"{key}.png")
        save_tensor_image(nir, output / "nir" / f"{key}.png")
        save_tensor_image(clean, output / "gt" / f"{key}.png")


if __name__ == "__main__":
    main()
