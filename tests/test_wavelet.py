import torch

from freqnir.wavelet import haar_dwt2, haar_idwt2


def test_haar_round_trip() -> None:
    image = torch.randn(2, 3, 16, 20)
    bands = haar_dwt2(image)
    reconstructed = haar_idwt2(*bands)
    torch.testing.assert_close(reconstructed, image)
