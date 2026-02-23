"""
Weight initialization following the original Pix2Pix paper.

From Isola et al. (2017): "weights are initialized from a Gaussian
distribution with mean 0 and standard deviation 0.02"
"""

import torch.nn as nn


def init_weights(m):
    """Initialize Conv and BatchNorm layers with N(0, 0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
