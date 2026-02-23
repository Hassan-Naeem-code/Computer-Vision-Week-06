"""PatchGAN Discriminator for Pix2Pix."""

import torch
import torch.nn as nn

from models.init_weights import init_weights


class Discriminator(nn.Module):
    """PatchGAN Discriminator.

    Takes concatenated input (satellite + map) and outputs a
    patch-level real/fake prediction map.
    """

    def __init__(self, in_channels=6):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: no batch norm
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: 1-channel prediction map
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    def forward(self, satellite, map_img):
        x = torch.cat((satellite, map_img), 1)
        return self.model(x)
