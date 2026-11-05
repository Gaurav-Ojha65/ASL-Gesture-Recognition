import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class LandmarkDataset(Dataset):
    """
    PyTorch Dataset for ASL landmark features.
    Loads data from a Parquet file containing pre-extracted and normalized 63-dimensional features.
    """

    def __init__(self, parquet_path: str):
        """
        Args:
            parquet_path: Path to the Parquet file containing landmark data.
        """
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self.df = pd.read_parquet(parquet_path)

        # Verify required columns exist
        required_columns = ['features', 'internal_class_index', 'semantic_label', 'source_path']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in {parquet_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            features: torch.Tensor of shape [63] and dtype torch.float32
            label: int, internal class index (0-27)
        """
        row = self.df.iloc[idx]

        # Features are stored as a numpy array in the Parquet file
        features = row['features']

        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)

        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Label is the internal class index
        label = int(row['internal_class_index'])

        # Ensure correct shape
        if features_tensor.shape != (63,):
            raise ValueError(f"Expected feature shape (63,), got {features_tensor.shape}")

        return features_tensor, label

    def get_source_path(self, idx: int) -> str:
        """Returns the original image path for the sample at the given index."""
        return self.df.iloc[idx]['source_path']

    def get_semantic_label(self, idx: int) -> str:
        """Returns the semantic label (e.g., 'A', 'B') for the sample at the given index."""
        return self.df.iloc[idx]['semantic_label']
