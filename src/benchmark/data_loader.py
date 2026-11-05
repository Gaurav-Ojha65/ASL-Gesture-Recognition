import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.benchmark.dataset import SignLanguageMNISTDataset

def get_dataloaders(train_csv_path, test_csv_path, batch_size=32, val_split=0.2, random_seed=42):
    """
    Creates DataLoaders for train, validation, and test sets.
    The training CSV is split into train and validation sets using stratified splitting.
    The test CSV is kept strictly separate.
    """
    # Load training data
    train_df = pd.read_csv(train_csv_path)

    # Stratified split for train and validation
    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=val_split,
        random_state=random_seed,
        stratify=train_df['label']
    )

    # DataLoaders
    train_dataset = SignLanguageMNISTDataset(dataframe=train_split_df)
    val_dataset = SignLanguageMNISTDataset(dataframe=val_split_df)
    test_dataset = SignLanguageMNISTDataset(csv_file=test_csv_path)

    # Shuffle only training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
