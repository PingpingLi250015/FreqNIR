from pathlib import Path

import numpy as np
from PIL import Image

from freqnir.data import PairedRGBNIRDataset


def test_paired_dataset(tmp_path: Path) -> None:
    for folder in ("noisy_rgb", "nir", "gt"):
        (tmp_path / folder).mkdir()
    rgb = np.full((20, 24, 3), 128, dtype=np.uint8)
    nir = np.full((20, 24), 64, dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "noisy_rgb" / "sample.png")
    Image.fromarray(nir).save(tmp_path / "nir" / "sample.png")
    Image.fromarray(rgb).save(tmp_path / "gt" / "sample.png")

    sample = PairedRGBNIRDataset(str(tmp_path), crop_size=16, augment=True)[0]
    assert sample["rgb"].shape == (3, 16, 16)
    assert sample["nir"].shape == (1, 16, 16)
    assert sample["target"].shape == (3, 16, 16)
