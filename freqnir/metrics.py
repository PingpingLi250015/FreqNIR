"""PSNR and SSIM evaluation metrics."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(prediction: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    return 10.0 * torch.log10((data_range**2) / mse.clamp_min(1e-12))


def _gaussian_window(size: int, sigma: float, channels: int, tensor: Tensor) -> Tensor:
    coordinate = torch.arange(size, device=tensor.device, dtype=tensor.dtype) - size // 2
    kernel = torch.exp(-(coordinate.square()) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).view(1, 1, size, size)
    return window.expand(channels, 1, size, size)


def ssim(prediction: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("SSIM inputs must have the same shape")
    channels = prediction.shape[1]
    window_size = min(11, prediction.shape[-2], prediction.shape[-1])
    if window_size % 2 == 0:
        window_size -= 1
    sigma = 1.5 * window_size / 11.0
    window = _gaussian_window(window_size, sigma, channels, prediction)
    padding = window_size // 2

    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = F.conv2d(prediction.square(), window, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return score.flatten(1).mean(dim=1)


def image_metrics(prediction: Tensor, target: Tensor) -> Dict[str, Tensor]:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    return {"psnr": psnr(prediction, target), "ssim": ssim(prediction, target)}
