import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model.mlp import ASLLandmarkMLP
from model.trainer import MLPTrainer
from model.data_loader import create_dataloaders
from landmarks.labels import SEMANTIC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

EPOCHS = 30
PATIENCE = 7
SEED = 42

def plot_curves(history, path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1'], label='Train')
    plt.plot(epochs, history['val_f1'], label='Validation')
    plt.title('Macro F1')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Internal Test Set')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(SEMANTIC_CLASSES))
    plt.xticks(tick_marks, SEMANTIC_CLASSES, rotation=45)
    plt.yticks(tick_marks, SEMANTIC_CLASSES)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dl, val_dl, test_dl = create_dataloaders(batch_size=64, seed=SEED)

    model = ASLLandmarkMLP()
    trainer = MLPTrainer(model, device=device, learning_rate=1e-3, weight_decay=1e-4, seed=SEED)

    best_val_f1 = -float('inf')
    best_epoch = -1
    best_val_metrics = {}
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}

    report_lines = []
    report_lines.append("# ASL MLP Training Report")
    report_lines.append(f"**Device:** {device}")
    report_lines.append(f"**Seed:** {SEED}")
    report_lines.append(f"**Dataset Sizes:** Train: {len(train_dl.dataset)}, Val: {len(val_dl.dataset)}, Test: {len(test_dl.dataset)}")
    report_lines.append("## Configuration\n- Loss: CrossEntropyLoss\n- Optimizer: AdamW\n- LR: 1e-3\n- Weight Decay: 1e-4\n- Batch Size: 64")
    report_lines.append("## Epochs")

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    checkpoint_path = 'models/asl_landmark_mlp_best.pt'

    start_time = time.time()
    total_samples = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_metrics = trainer.train_epoch(train_dl)
        val_metrics = trainer.evaluate(val_dl)

        # Need train accuracy, f1 etc. so we run evaluate on train_dl.
        # But running evaluation on full train_dl takes time, let's just do it.
        train_eval_metrics = trainer.evaluate(train_dl)

        epoch_dur = time.time() - epoch_start
        total_samples += len(train_dl.dataset)

        history['train_loss'].append(train_eval_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_eval_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_eval_metrics['macro_f1'])
        history['val_f1'].append(val_metrics['macro_f1'])

        val_f1 = val_metrics['macro_f1']
        is_best = False

        if val_f1 > best_val_f1 + 1e-5: # Small positive delta
            best_val_f1 = val_f1
            best_epoch = epoch
            best_val_metrics = val_metrics
            trainer.epoch = epoch
            trainer.best_val_metric = best_val_f1
            trainer.save_checkpoint(checkpoint_path)
            epochs_no_improve = 0
            is_best = True
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:02d} | Train Loss: {train_eval_metrics['loss']:.4f} F1: {train_eval_metrics['macro_f1']:.4f} | Val Loss: {val_metrics['loss']:.4f} F1: {val_f1:.4f} | {'*BEST*' if is_best else ''}")

        report_lines.append(f"### Epoch {epoch}")
        report_lines.append(f"- Duration: {epoch_dur:.2f}s")
        report_lines.append(f"- Train: Loss {train_eval_metrics['loss']:.4f}, Acc {train_eval_metrics['accuracy']:.4f}, F1 {train_eval_metrics['macro_f1']:.4f}")
        report_lines.append(f"- Val: Loss {val_metrics['loss']:.4f}, Acc {val_metrics['accuracy']:.4f}, F1 {val_metrics['macro_f1']:.4f}")
        report_lines.append(f"- New Best: {is_best}")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            report_lines.append(f"\n**Early stopping triggered at epoch {epoch}**")
            break

    total_time = time.time() - start_time
    throughput = total_samples / total_time

    report_lines.append("\n## Final Summary")
    report_lines.append(f"- **Runtime:** {total_time:.2f}s")
    report_lines.append(f"- **Throughput:** {throughput:.2f} samples/s")
    report_lines.append(f"- **Best Epoch:** {best_epoch}")
    report_lines.append(f"- **Best Val Loss:** {best_val_metrics['loss']:.4f}")
    report_lines.append(f"- **Best Val Acc:** {best_val_metrics['accuracy']:.4f}")
    report_lines.append(f"- **Best Val F1:** {best_val_metrics['macro_f1']:.4f}")

    with open('reports/asl_mlp_training_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    plot_curves(history, 'reports/asl_mlp_training_curves.png')

    print("\nTraining complete. Running internal-test evaluation...")

    # Internal Test Evaluation
    fresh_model = ASLLandmarkMLP()
    fresh_trainer = MLPTrainer(fresh_model, device=device)
    fresh_trainer.load_checkpoint(checkpoint_path)

    # Collect predictions
    fresh_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_dl:
            features = features.to(device)
            logits = fresh_model(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute Classification Report
    cr = classification_report(all_labels, all_preds, target_names=SEMANTIC_CLASSES, zero_division=0)

    cr_lines = []
    cr_lines.append("# Internal Test Set Classification Report")
    cr_lines.append(f"Model: {checkpoint_path}")
    cr_lines.append("## Evaluation")
    cr_lines.append("```text")
    cr_lines.append(cr)
    cr_lines.append("```")
    cr_lines.append("\n> **Important Note:** Per-class metrics based on very small support have greater statistical uncertainty. Do not convert this into claims about why those classes behave differently.")

    with open('reports/asl_mlp_classification_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(cr_lines))

    plot_confusion_matrix(all_labels, all_preds, 'reports/asl_mlp_confusion_matrix.png')
    print("Done.")

if __name__ == '__main__':
    # suppress sklearn undefined metric warning
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
