"""Plotting and visualization utilities."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] for display."""
    return tensor * 0.5 + 0.5


def show_sample(dataset, save_path):
    """Save a sample satellite-map pair from the dataset."""
    satellite, map_img = dataset[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    sat_display = denormalize(satellite).permute(1, 2, 0).numpy()
    map_display = denormalize(map_img).permute(1, 2, 0).numpy()

    axes[0].imshow(sat_display)
    axes[0].set_title("Satellite Domain")
    axes[0].axis("off")
    axes[1].imshow(map_display)
    axes[1].set_title("Map Domain")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample pair saved to: {save_path}")


def plot_losses(g_losses, d_losses, save_path):
    """Plot generator and discriminator training loss curves."""
    epochs = range(1, len(g_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, g_losses, label="Generator Loss")
    plt.plot(epochs, d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pix2Pix Training Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def plot_results(generator, val_dataset, device, num_samples, save_path):
    """Generate side-by-side translation results on validation data."""
    generator.eval()
    num_samples = min(num_samples, len(val_dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # Handle single-sample case (axes is 1D)
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i in range(num_samples):
            sat_img, map_img = val_dataset[i]
            sat_input = sat_img.unsqueeze(0).to(device)
            translated = generator(sat_input).cpu().squeeze(0)

            sat_display = denormalize(sat_img).permute(1, 2, 0).numpy()
            map_display = denormalize(map_img).permute(1, 2, 0).numpy()
            translated_display = np.clip(
                denormalize(translated).permute(1, 2, 0).numpy(), 0, 1
            )

            axes[i, 0].imshow(sat_display)
            axes[i, 0].set_title("Input (Satellite)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(translated_display)
            axes[i, 1].set_title("Generated (Map)")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(map_display)
            axes[i, 2].set_title("Ground Truth (Map)")
            axes[i, 2].axis("off")

    plt.suptitle("Pix2Pix: Satellite to Map Translation Results", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Translation results saved to: {save_path}")
