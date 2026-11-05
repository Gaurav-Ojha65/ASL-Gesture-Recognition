import torch
from torch.utils.data import DataLoader
from typing import Tuple
from .landmark_dataset import LandmarkDataset
import os

def create_dataloaders(
    train_parquet_path: str = 'data/processed/landmarks/full/landmarks_train.parquet',
    val_parquet_path: str = 'data/processed/landmarks/full/landmarks_validation.parquet',
    test_parquet_path: str = 'data/processed/landmarks/full/landmarks_internal_test.parquet',
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and internal test sets.

    Args:
        train_parquet_path: Path to the training Parquet file.
        val_parquet_path: Path to the validation Parquet file.
        test_parquet_path: Path to the internal test Parquet file.
        batch_size: Batch size for all loaders.
        num_workers: Number of workers for data loading.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set seed for reproducibility in DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = LandmarkDataset(train_parquet_path)
    val_dataset = LandmarkDataset(val_parquet_path)
    test_dataset = LandmarkDataset(test_parquet_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
