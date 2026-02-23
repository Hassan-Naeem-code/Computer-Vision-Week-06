"""Satellite-to-Map paired image dataset and data loading utilities."""

import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class SatelliteMapDataset(Dataset):
    """Paired satellite-to-map dataset.

    Each source image contains the satellite photo on the left half
    and the corresponding map on the right half. This dataset splits
    them into separate tensors.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        w, h = image.size
        satellite = image.crop((0, 0, w // 2, h))
        map_img = image.crop((w // 2, 0, w, h))

        if self.transform:
            satellite = self.transform(satellite)
            map_img = self.transform(map_img)

        return satellite, map_img


def get_transforms(image_size=256):
    """Build the image transform pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataloaders(config):
    """Create train and validation DataLoaders from config."""
    transform = get_transforms(config.image_size)

    train_dataset = SatelliteMapDataset(config.train_dir, transform=transform)
    val_dataset = SatelliteMapDataset(config.val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
