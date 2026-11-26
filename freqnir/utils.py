"""Configuration, checkpoint, image, and tiled-inference helpers."""

import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor, nn


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    model = config.get("model", config)
    return {
        "base_channels": int(model.get("base_channels", 30)),
        "block_depths": tuple(model.get("block_depths", [2, 2, 2, 2, 2, 2])),
        "bias_window": int(model.get("bias_window", 16)),
        "attention_chunk_size": int(model.get("attention_chunk_size", 512)),
        "ffn_expansion": float(model.get("ffn_expansion", 2.0)),
    }


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def load_checkpoint(
    model: nn.Module, path: str, map_location: str = "cpu", strict: bool = True
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    clean_state = {
        (key[7:] if key.startswith("module.") else key): value for key, value in state.items()
    }
    unwrap_model(model).load_state_dict(clean_state, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {"model": state}


def save_tensor_image(tensor: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
    array = np.round(array * 255.0).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


def _tile_starts(length: int, tile: int, overlap: int) -> Sequence[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile overlap must be smaller than tile size")
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tiled_forward(
    model: nn.Module, rgb: Tensor, nir: Tensor, tile_size: int = 0, overlap: int = 32
) -> Tensor:
    """Run full-frame inference or overlap-average spatial tiles."""
    if tile_size <= 0 or max(rgb.shape[-2:]) <= tile_size:
        return model(rgb, nir)

    batch, _, height, width = rgb.shape
    if batch != 1:
        raise ValueError("Tiled inference currently expects batch size 1")
    output = torch.zeros_like(rgb)
    weight = torch.zeros_like(rgb[:, :1])
    for top in _tile_starts(height, tile_size, overlap):
        for left in _tile_starts(width, tile_size, overlap):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            rgb_tile = rgb[..., top:bottom, left:right]
            nir_tile = nir[..., top:bottom, left:right]
            prediction = model(rgb_tile, nir_tile)
            output[..., top:bottom, left:right] += prediction
            weight[..., top:bottom, left:right] += 1.0
    return output / weight.clamp_min(1.0)
