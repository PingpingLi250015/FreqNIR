"""Paired RGB-NIR datasets with synchronized cropping and augmentation."""

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def _normalize(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    array = array.astype(np.float32)
    maximum = float(np.nanmax(array)) if array.size else 1.0
    if maximum <= 1.0:
        return np.nan_to_num(array).clip(0.0, 1.0)
    for scale in (255.0, 1023.0, 4095.0, 65535.0):
        if maximum <= scale:
            return np.nan_to_num(array / scale).clip(0.0, 1.0)
    raise ValueError(f"Cannot infer normalization for floating-point data with maximum {maximum}")


def load_image(path: Path, channels: int) -> Tensor:
    """Load an image or numeric NPY array into a normalized CHW tensor."""
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.dtype.names:
            raise ValueError(
                f"Structured raw array {path} must be converted to an RGB/NIR image first"
            )
    else:
        with Image.open(path) as image:
            image = image.convert("RGB" if channels == 3 else "L")
            array = np.asarray(image)

    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC array at {path}, got {array.shape}")
    if array.shape[0] <= 4 and array.shape[-1] > 4:
        array = np.moveaxis(array, 0, -1)

    if channels == 1 and array.shape[-1] != 1:
        array = array.mean(axis=-1, keepdims=True)
    elif channels == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] != channels:
        raise ValueError(f"Expected {channels} channels at {path}, got {array.shape[-1]}")

    array = np.ascontiguousarray(_normalize(array).transpose(2, 0, 1))
    return torch.from_numpy(array)


def discover_images(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    found: Dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.relative_to(directory).with_suffix("").as_posix()
        if key in found:
            raise ValueError(f"Duplicate sample key '{key}' in {directory}")
        found[key] = path
    return found


def _pad_for_crop(tensors: Sequence[Tensor], crop_size: int) -> List[Tensor]:
    height, width = tensors[0].shape[-2:]
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if not pad_h and not pad_w:
        return list(tensors)
    mode = "reflect" if min(height, width) > 1 else "replicate"
    return [F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode) for tensor in tensors]


def synchronized_transform(
    rgb: Tensor, nir: Tensor, target: Tensor, crop_size: int, augment: bool
) -> Tuple[Tensor, Tensor, Tensor]:
    tensors = _pad_for_crop((rgb, nir, target), crop_size)
    height, width = tensors[0].shape[-2:]
    top = int(torch.randint(0, height - crop_size + 1, ()).item())
    left = int(torch.randint(0, width - crop_size + 1, ()).item())
    tensors = [t[..., top : top + crop_size, left : left + crop_size] for t in tensors]

    if augment:
        if bool(torch.randint(0, 2, ())):
            tensors = [t.flip(-1) for t in tensors]
        if bool(torch.randint(0, 2, ())):
            tensors = [t.flip(-2) for t in tensors]
        rotations = int(torch.randint(0, 4, ()).item())
        tensors = [torch.rot90(t, rotations, (-2, -1)) for t in tensors]
    return tensors[0], tensors[1], tensors[2]


class PairedRGBNIRDataset(Dataset):
    """Read ``noisy_rgb``, ``nir`` and ``gt`` folders paired by relative stem."""

    def __init__(
        self,
        root: str,
        crop_size: Optional[int] = None,
        augment: bool = False,
        require_target: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.crop_size = crop_size
        self.augment = augment
        self.rgb = discover_images(self.root / "noisy_rgb")
        self.nir = discover_images(self.root / "nir")
        self.target = discover_images(self.root / "gt") if require_target else {}

        common = set(self.rgb) & set(self.nir)
        if require_target:
            common &= set(self.target)
        self.keys = sorted(common)
        if not self.keys:
            raise RuntimeError(f"No paired samples found below {self.root}")

        expected = set(self.rgb) | set(self.nir)
        if require_target:
            expected |= set(self.target)
        missing = sorted(expected - common)
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Unpaired files below {self.root}: {preview}")
        self.require_target = require_target

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        key = self.keys[index]
        rgb = load_image(self.rgb[key], 3)
        nir = load_image(self.nir[key], 1)
        if rgb.shape[-2:] != nir.shape[-2:]:
            raise ValueError(f"Spatial mismatch for '{key}': {rgb.shape} vs {nir.shape}")

        sample: Dict[str, object] = {"rgb": rgb, "nir": nir, "name": key}
        if self.require_target:
            target = load_image(self.target[key], 3)
            if target.shape[-2:] != rgb.shape[-2:]:
                raise ValueError(f"Target mismatch for '{key}': {target.shape} vs {rgb.shape}")
            if self.crop_size is not None:
                rgb, nir, target = synchronized_transform(
                    rgb, nir, target, self.crop_size, self.augment
                )
            sample.update({"rgb": rgb, "nir": nir, "target": target})
        return sample
