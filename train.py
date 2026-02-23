"""
Pix2Pix GAN Training Script
Week 6 Assignment - Computer Vision

Usage:
    python train.py                                    # train with defaults
    python train.py --epochs 20 --lr 0.001             # override via CLI
    python train.py --resume_from checkpoints/checkpoint_epoch_5.pt
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from models import Generator, Discriminator
from data.dataset import get_dataloaders
from utils import set_seed, save_checkpoint, show_sample, plot_losses, plot_results


def train(config):
    # --- Setup ---
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device)
    print(f"Configuration:\n{config}\n")

    # --- Data ---
    train_loader, val_loader = get_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    show_sample(train_loader.dataset, os.path.join(config.output_dir, "sample_pair.png"))

    # --- Models ---
    generator = Generator(config.in_channels, config.out_channels).to(device)
    discriminator = Discriminator(in_channels=config.in_channels * 2).to(device)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # --- Losses and Optimizers ---
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    # --- Resume from checkpoint if requested ---
    start_epoch = 0
    g_losses, d_losses = [], []
    if config.resume_from:
        from utils import load_checkpoint
        start_epoch, g_losses, d_losses = load_checkpoint(
            config.resume_from, generator, discriminator,
            optimizer_G, optimizer_D, device=config.device,
        )
        print(f"Resumed from epoch {start_epoch}")

    # --- Training Loop ---
    print("\n--- Starting Training ---\n")

    for epoch in range(start_epoch, config.epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for sat_images, map_images in pbar:
            sat_images = sat_images.to(device)
            map_images = map_images.to(device)

            # Generate fake map images from satellite images
            fake_map = generator(sat_images)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            pred_real = discriminator(sat_images, map_images)
            real_loss = adversarial_loss(pred_real, torch.ones_like(pred_real))

            pred_fake = discriminator(sat_images, fake_map.detach())
            fake_loss = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            pred_fake_for_g = discriminator(sat_images, fake_map)
            g_adv_loss = adversarial_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            g_l1_loss = l1_loss(fake_map, map_images)
            total_g_loss = g_adv_loss + config.lambda_l1 * g_l1_loss

            total_g_loss.backward()
            optimizer_G.step()

            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += d_loss.item()
            pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{total_g_loss.item():.4f}")

        # --- Epoch Summary ---
        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        print(f"Epoch [{epoch + 1}/{config.epochs}] - D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}")

        # --- Periodic Checkpoint ---
        if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.epochs:
            path = save_checkpoint(
                generator, discriminator, optimizer_G, optimizer_D,
                epoch + 1, g_losses, d_losses, config,
            )
            print(f"  Checkpoint saved: {path}")

    print("\n--- Training Complete ---\n")

    # --- Post-training Outputs ---
    plot_losses(g_losses, d_losses, os.path.join(config.output_dir, "training_losses.png"))
    plot_results(
        generator, val_loader.dataset, device,
        config.num_samples, os.path.join(config.output_dir, "translation_results.png"),
    )

    # Save final model weights (lightweight, for inference only)
    torch.save(generator.state_dict(), os.path.join(config.output_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(config.output_dir, "discriminator.pth"))
    print(f"Final models saved to {config.output_dir}/")


if __name__ == "__main__":
    config = Config.from_args()
    train(config)
