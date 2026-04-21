from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _subset_if_needed(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_dataloaders(
    data_dir: str,
    input_resolution: int,
    batch_size: int,
    train_subset: int | None,
    val_subset: int | None,
    test_subset: int | None,
    num_workers: int,
    seed: int,
) -> DataBundle:
    transform = transforms.Compose(
        [
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=split_gen)

    train_dataset = _subset_if_needed(train_dataset, train_subset, seed)
    val_dataset = _subset_if_needed(val_dataset, val_subset, seed + 1)
    test_dataset = _subset_if_needed(test_dataset, test_subset, seed + 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
