"""Checkpoint save/load utilities for training resumption."""

import os
import torch


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D,
                    epoch, g_losses, d_losses, config):
    """Save full training state to resume later."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "g_losses": g_losses,
        "d_losses": d_losses,
    }, path)
    return path


def load_checkpoint(path, generator, discriminator,
                    optimizer_G=None, optimizer_D=None, device="cpu"):
    """Load checkpoint. Returns (epoch, g_losses, d_losses).

    If optimizers are None (evaluation mode), only model weights are loaded.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if optimizer_G is not None and optimizer_D is not None:
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    return checkpoint["epoch"], checkpoint["g_losses"], checkpoint["d_losses"]
