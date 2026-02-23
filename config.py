"""
Pix2Pix GAN Configuration

All hyperparameters and paths in one place. Override via CLI:
    python train.py --epochs 20 --lr 0.001 --batch_size 8
"""

from dataclasses import dataclass
import argparse
import torch


@dataclass
class Config:
    # Paths
    train_dir: str = "./data/maps/train"
    val_dir: str = "./data/maps/val"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 16
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0

    # Model
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 3

    # System
    device: str = ""
    num_workers: int = 2
    seed: int = 42

    # Checkpointing
    save_every: int = 5
    resume_from: str = ""

    # Evaluation
    num_samples: int = 5

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def from_args() -> "Config":
        """Parse CLI arguments, overriding dataclass defaults."""
        config = Config()
        parser = argparse.ArgumentParser(description="Pix2Pix GAN")
        for field_name, field_obj in config.__dataclass_fields__.items():
            default_val = getattr(config, field_name)
            parser.add_argument(
                f"--{field_name}",
                type=type(default_val) if default_val != "" else str,
                default=default_val,
            )
        args = parser.parse_args()
        return Config(**vars(args))
