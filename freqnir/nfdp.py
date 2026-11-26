"""NIR Frequency Domain Prior (NFDP) extraction network."""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .wavelet import haar_dwt2, haar_idwt2


class SobelGradient(nn.Module):
    """Channel-wise Sobel derivatives used by differential fusion."""

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_x.t().view(1, 1, 3, 3))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        channels = x.shape[1]
        weight_x = self.kernel_x.expand(channels, 1, 3, 3)
        weight_y = self.kernel_y.expand(channels, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return (
            F.conv2d(x, weight_x, groups=channels),
            F.conv2d(x, weight_y, groups=channels),
        )


class DifferentialSpectrumProcessor(nn.Module):
    """Learn weights/interpolation vectors and apply Eq. (6)."""

    def __init__(self, channels: int, kernel_size: int, amplitude: bool) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.predict = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size, padding=padding),
            nn.SELU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
        )
        self.gradient = SobelGradient()
        self.residual_scale = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))
        self.amplitude = amplitude

        # NFDP starts as an identity frequency transform and learns the correction.
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, spectrum: Tensor) -> Tensor:
        weight, interpolation = self.predict(spectrum).chunk(2, dim=1)
        weight = torch.sigmoid(weight)
        wx, wy = self.gradient(weight)
        vx, vy = self.gradient(interpolation)
        differential = wx * vx + wy * vy
        residual = F.softplus(self.residual_scale) * spectrum
        fused = differential + residual
        if self.amplitude:
            fused = fused.clamp_min(1e-8)
        return fused


class FourierLowFrequencyUnit(nn.Module):
    """Amplitude/phase refinement of a DWT low-frequency band."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Eq. (5): point-wise amplitude processing and spatial phase processing.
        self.amplitude = DifferentialSpectrumProcessor(channels, 1, amplitude=True)
        self.phase = DifferentialSpectrumProcessor(channels, 3, amplitude=False)

    def forward(self, x: Tensor) -> Tensor:
        spectrum = torch.fft.fft2(x, norm="ortho")
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        amplitude = self.amplitude(amplitude)
        phase = self.phase(phase)
        refined = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
        return torch.fft.ifft2(refined, norm="ortho").real


class NFDPStage(nn.Module):
    """One DWT-frequency-IDWT stage followed by a scale transition."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.low_projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
        )
        self.fourier = FourierLowFrequencyUnit(in_channels)
        self.transition = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.low_to_rgb = nn.Conv2d(in_channels, 3, 1)
        self.prior_to_rgb = nn.Conv2d(out_channels, 3, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ll, lh, hl, hh = haar_dwt2(x)
        low = self.fourier(self.low_projection(ll))
        reconstructed = haar_idwt2(low, lh, hl, hh)
        prior = self.transition(reconstructed)
        return prior, self.low_to_rgb(low), self.prior_to_rgb(prior)


class NFDP(nn.Module):
    """Three-stage multi-scale NIR prior extractor from Sec. 3.2."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, base_channels, 3, padding=1)
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stages = nn.ModuleList([NFDPStage(channels[i], channels[i + 1]) for i in range(3)])

    def forward(self, nir: Tensor) -> Dict[str, List[Tensor]]:
        x = self.stem(nir)
        priors: List[Tensor] = []
        low_predictions: List[Tensor] = []
        prior_predictions: List[Tensor] = []
        for stage in self.stages:
            x, low_rgb, prior_rgb = stage(x)
            priors.append(x)
            low_predictions.append(low_rgb)
            prior_predictions.append(prior_rgb)
        return {
            "priors": priors,
            "low_predictions": low_predictions,
            "prior_predictions": prior_predictions,
        }
