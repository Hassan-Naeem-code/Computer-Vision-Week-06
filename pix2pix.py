"""
Pix2Pix GAN for Satellite-to-Map Image Translation
Week 6 Assignment - Computer Vision
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Step 3: Load and Preprocess the Satellite-to-Map Dataset
# ============================================================

class SatelliteMapDataset(Dataset):
    """
    The maps dataset contains paired images where the left half is the
    satellite image and the right half is the corresponding map.
    This dataset splits each image into its satellite and map components.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # The maps dataset has paired images side by side
        # Left half = satellite image, Right half = map image
        w, h = image.size
        satellite = image.crop((0, 0, w // 2, h))
        map_img = image.crop((w // 2, 0, w, h))

        if self.transform:
            satellite = self.transform(satellite)
            map_img = self.transform(map_img)

        return satellite, map_img


# Define transformations (resize, normalize, convert to tensors)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def show_sample(dataset, title_prefix="Sample"):
    """Display a sample satellite-map image pair."""
    satellite, map_img = dataset[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # Denormalize from [-1, 1] to [0, 1]
    sat_display = satellite.permute(1, 2, 0) * 0.5 + 0.5
    map_display = map_img.permute(1, 2, 0) * 0.5 + 0.5
    axes[0].imshow(sat_display.numpy())
    axes[0].set_title(f"{title_prefix} - Satellite Domain")
    axes[0].axis("off")
    axes[1].imshow(map_display.numpy())
    axes[1].set_title(f"{title_prefix} - Map Domain")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("sample_pair.png", dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Step 4: Implement Pix2Pix Model
# ============================================================

# --- Generator (U-Net style) ---

class UNetDown(nn.Module):
    """Downsampling block for U-Net encoder."""
    def __init__(self, in_channels, out_channels, normalize=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block for U-Net decoder with skip connections."""
    def __init__(self, in_channels, out_channels, dropout=False):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    """U-Net based Generator for Pix2Pix."""
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)   # 256 -> 128
        self.down2 = UNetDown(64, 128)                             # 128 -> 64
        self.down3 = UNetDown(128, 256)                            # 64  -> 32
        self.down4 = UNetDown(256, 512)                            # 32  -> 16
        self.down5 = UNetDown(512, 512)                            # 16  -> 8
        self.down6 = UNetDown(512, 512)                            # 8   -> 4
        self.down7 = UNetDown(512, 512)                            # 4   -> 2
        self.down8 = UNetDown(512, 512, normalize=False)           # 2   -> 1

        # Decoder (upsampling with skip connections)
        self.up1 = UNetUp(512, 512, dropout=True)                  # 1   -> 2
        self.up2 = UNetUp(1024, 512, dropout=True)                 # 2   -> 4
        self.up3 = UNetUp(1024, 512, dropout=True)                 # 4   -> 8
        self.up4 = UNetUp(1024, 512)                               # 8   -> 16
        self.up5 = UNetUp(1024, 256)                               # 16  -> 32
        self.up6 = UNetUp(512, 128)                                # 32  -> 64
        self.up7 = UNetUp(256, 64)                                 # 64  -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()                                              # 128 -> 256
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# --- Discriminator (PatchGAN style) ---

class Discriminator(nn.Module):
    """PatchGAN Discriminator for Pix2Pix.
    Takes concatenated input (satellite + map) and outputs a patch-level prediction.
    """
    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__()
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
            nn.Sigmoid()
        )

    def forward(self, satellite, map_img):
        # Concatenate satellite and map images along channel dimension
        x = torch.cat((satellite, map_img), 1)
        return self.model(x)


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] for display."""
    return tensor * 0.5 + 0.5


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":

    # --- Load datasets ---
    train_dataset = SatelliteMapDataset(root_dir="./data/maps/train", transform=transform)
    val_dataset = SatelliteMapDataset(root_dir="./data/maps/val", transform=transform)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    show_sample(train_dataset)

    # --- Initialize models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    G_sat2map = Generator().to(device)
    D_map = Discriminator().to(device)

    print(f"Generator parameters: {sum(p.numel() for p in G_sat2map.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in D_map.parameters()):,}")

    # ============================================================
    # Step 5: Train Pix2Pix
    # ============================================================

    # Define loss functions
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    # Optimizers
    lr = 0.0002
    beta1 = 0.5
    optimizer_G = optim.Adam(G_sat2map.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D_map.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop
    epochs = 10
    lambda_l1 = 100  # Weight for L1 loss (as in the original Pix2Pix paper)

    g_losses = []
    d_losses = []

    print("\n--- Starting Training ---\n")

    for epoch in range(epochs):
        G_sat2map.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for sat_images, map_images in train_loader:
            sat_images = sat_images.to(device)
            map_images = map_images.to(device)

            # Generate fake map images from satellite images
            fake_map = G_sat2map(sat_images)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real loss: discriminator should predict real for real pairs
            pred_real = D_map(sat_images, map_images)
            real_labels = torch.ones_like(pred_real)
            real_loss = adversarial_loss(pred_real, real_labels)

            # Fake loss: discriminator should predict fake for generated pairs
            pred_fake = D_map(sat_images, fake_map.detach())
            fake_labels = torch.zeros_like(pred_fake)
            fake_loss = adversarial_loss(pred_fake, fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            # Adversarial loss: generator wants discriminator to predict real
            pred_fake_for_g = D_map(sat_images, fake_map)
            g_adv_loss = adversarial_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

            # L1 loss: encourage generated image to be close to ground truth
            g_l1_loss = l1_loss(fake_map, map_images)

            total_g_loss = g_adv_loss + lambda_l1 * g_l1_loss
            total_g_loss.backward()
            optimizer_G.step()

            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

    print("\n--- Training Complete ---\n")

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), g_losses, label="Generator Loss")
    plt.plot(range(1, epochs + 1), d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pix2Pix Training Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_losses.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ============================================================
    # Step 6: Evaluate and Visualize Translated Images
    # ============================================================

    G_sat2map.eval()

    # Generate translated images from validation set
    num_samples = 5
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            sat_img, map_img = val_dataset[i]
            sat_input = sat_img.unsqueeze(0).to(device)
            translated = G_sat2map(sat_input).cpu().squeeze(0)

            # Denormalize for display
            sat_display = denormalize(sat_img).permute(1, 2, 0).numpy()
            map_display = denormalize(map_img).permute(1, 2, 0).numpy()
            translated_display = denormalize(translated).permute(1, 2, 0).numpy()
            translated_display = np.clip(translated_display, 0, 1)

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
    plt.savefig("translation_results.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Save the trained models
    torch.save(G_sat2map.state_dict(), "generator.pth")
    torch.save(D_map.state_dict(), "discriminator.pth")
    print("Models saved: generator.pth, discriminator.pth")
