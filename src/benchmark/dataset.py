import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SignLanguageMNISTDataset(Dataset):
    """
    PyTorch Dataset for Sign Language MNIST.
    """
    def __init__(self, csv_file=None, dataframe=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            dataframe (pd.DataFrame): DataFrame containing the data (used for train/val split).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert (csv_file is not None) ^ (dataframe is not None), "Provide exactly one of csv_file or dataframe"

        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        else:
            self.data = dataframe.reset_index(drop=True)

        self.transform = transform

        # Original labels are 0-8, 10-24. CrossEntropyLoss expects contiguous 0-23.
        # We create a mapping.
        self.original_labels = sorted(self.data['label'].unique().tolist())
        # The expected complete set of labels in the dataset
        expected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        # Verify no unexpected labels exist
        for l in self.original_labels:
            assert l in expected_labels, f"Unexpected label {l} found in dataset"

        self.label_to_index = {label: idx for idx, label in enumerate(expected_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get original label and map to contiguous index [0, 23]
        original_label = self.data.iloc[idx, 0]
        mapped_label = self.label_to_index[original_label]

        # Get pixels (784 columns), reshape to 28x28, shape to [1, 28, 28]
        # Data is in uint8 (0-255), we normalize it to float32 (0.0-1.0)
        pixels = self.data.iloc[idx, 1:].values.astype(np.float32)
        image = pixels.reshape((1, 28, 28))
        image = image / 255.0

        # Convert to torch tensor
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.tensor(mapped_label, dtype=torch.long)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_tensor
