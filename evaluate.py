"""
Pix2Pix GAN Evaluation Script
Week 6 Assignment - Computer Vision

Usage:
    python evaluate.py                                              # uses default weights
    python evaluate.py --resume_from outputs/generator.pth          # specify weights
    python evaluate.py --num_samples 10                             # more samples
"""

import os

import torch

from config import Config
from models import Generator
from data.dataset import get_dataloaders
from utils import plot_results


def evaluate(config):
    device = torch.device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Data ---
    _, val_loader = get_dataloaders(config)
    print(f"Validation samples: {len(val_loader.dataset)}")

    # --- Load Generator ---
    generator = Generator(config.in_channels, config.out_channels).to(device)

    weights_path = config.resume_from
    if not weights_path:
        weights_path = os.path.join(config.output_dir, "generator.pth")

    state = torch.load(weights_path, map_location=device, weights_only=False)
    # Handle both full checkpoint and raw state_dict formats
    if isinstance(state, dict) and "generator_state_dict" in state:
        generator.load_state_dict(state["generator_state_dict"])
    else:
        generator.load_state_dict(state)
    print(f"Loaded generator from: {weights_path}")

    # --- Generate Results ---
    save_path = os.path.join(config.output_dir, "evaluation_results.png")
    plot_results(generator, val_loader.dataset, device, config.num_samples, save_path)


if __name__ == "__main__":
    config = Config.from_args()
    evaluate(config)
