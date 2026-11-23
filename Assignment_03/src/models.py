"""ChannelNet networks (paper Sec. III-B).

SRCNN  (Dong et al. 2014) - super-resolution stage
    input : LS estimate at the pilots, linearly interpolated to 72 x 14
            (SRCNN always starts from an interpolated low-res image)
    conv 9x9, 64 filters + ReLU   -> patch extraction
    conv 1x1, 32 filters + ReLU   -> non-linear mapping
    conv 5x5,  1 filter           -> reconstruction
    'same' padding keeps the 72 x 14 size.

DnCNN  (Zhang et al. 2017) - image restoration / denoising stage
    conv 3x3, 64 + ReLU
    18 x [conv 3x3, 64 + BatchNorm + ReLU]
    conv 3x3, 1
    Residual learning: the network predicts the NOISE n, output = Z - n.

Real and imaginary parts are processed as separate 1-channel images by the
same network (paper: "once the real-part and then the imaginary-part").
"""
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):          # x: (B, 1, 72, 14)
        return self.net(x)


class DnCNN(nn.Module):
    def __init__(self, depth: int = 20, channels: int = 64):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):          # residual learning: H_hat = Z - noise_hat
        return z - self.net(z)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
