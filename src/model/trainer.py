import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import json
from typing import Dict, Any, Optional

from .mlp import ASLLandmarkMLP
from landmarks.labels import SEMANTIC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

class MLPTrainer:
    """
    Training infrastructure for the ASL Landmark MLP.
    Supports train mode, evaluation mode, loss calculation, metrics, and checkpointing.
    """
    def __init__(
        self,
        model: ASLLandmarkMLP,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42
    ):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.epoch = 0
        self.best_val_metric = -float('inf')

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Runs one epoch of training and returns average loss."""
        self.model.train()
        total_loss = 0.0

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)

        return {'loss': total_loss / len(dataloader.dataset)}

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Runs evaluation and computes loss, accuracy, and macro P/R/F1."""
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * features.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        return {
            'loss': total_loss / len(dataloader.dataset),
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1
        }

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Saves a checkpoint containing model state, optimizer state, and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'model_architecture': 'ASLLandmarkMLP',
            'num_classes': 28,
            'input_dim': 63,
            'class_to_index': CLASS_TO_IDX,
            'index_to_class': IDX_TO_CLASS,
            'best_val_metric': self.best_val_metric,
            'random_seed': self.seed,
            'training_hyperparameters': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'loss_function': 'CrossEntropyLoss',
                'optimizer': 'AdamW'
            }
        }

        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Loads a checkpoint into the current model and optimizer."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint.get('best_val_metric', -float('inf'))
        return checkpoint
