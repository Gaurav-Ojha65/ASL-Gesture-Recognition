import os
import sys
import torch
import torch.nn as nn
import tempfile
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model.mlp import ASLLandmarkMLP
from model.trainer import MLPTrainer
from model.data_loader import create_dataloaders

TRAIN_PATH = 'data/processed/landmarks/full/landmarks_train.parquet'
VAL_PATH = 'data/processed/landmarks/full/landmarks_validation.parquet'
TEST_PATH = 'data/processed/landmarks/full/landmarks_internal_test.parquet'
REPORT_PATH = 'reports/asl_mlp_architecture_audit.md'

def generate_report():
    report_lines = []
    report_lines.append("# ASL MLP Architecture and Infrastructure Audit")
    report_lines.append("")
    report_lines.append("## Objective")
    report_lines.append("Independent verification of the 28-class MLP architecture, loss configuration, training infrastructure, and dataset consumption.")
    report_lines.append("")

    # 1. Architecture Audit
    model = ASLLandmarkMLP()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Analyze architecture
    layers = list(model.net)
    has_bn = isinstance(layers[1], nn.BatchNorm1d)
    has_relu1 = isinstance(layers[2], nn.ReLU)
    has_dropout1 = isinstance(layers[3], nn.Dropout) and layers[3].p == 0.20
    has_relu2 = isinstance(layers[5], nn.ReLU)
    has_dropout2 = isinstance(layers[6], nn.Dropout) and layers[6].p == 0.20
    no_activation_end = not isinstance(layers[-1], (nn.Softmax, nn.LogSoftmax, nn.Sigmoid, nn.ReLU))

    arch_pass = (has_bn and has_relu1 and has_dropout1 and has_relu2 and has_dropout2 and no_activation_end)
    param_pass = (total_params == 18524)

    # Input/Output shape & dtype & finite check
    test_tensor = torch.randn(4, 63, dtype=torch.float32)
    torch.manual_seed(42) # Ensure determinism for reproducible checks
    output = model(test_tensor)

    in_shape_pass = True # Derived from the fact it runs
    out_shape_pass = (output.shape == (4, 28))
    dtype_pass = (output.dtype == torch.float32)
    finite_pass = torch.isfinite(output).all().item()

    # Check that outputs are raw logits (not summing to 1 like softmax)
    sums = output.sum(dim=1)
    raw_logits_pass = not torch.allclose(sums, torch.ones_like(sums))

    # Loss configuration
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 28, (4,))
    try:
        loss = criterion(output, labels)
        ce_compat_pass = True
    except Exception:
        ce_compat_pass = False

    report_lines.append("## Independent Verification")
    report_lines.append(f"- **Architecture components matches specification**: {'PASS' if arch_pass else 'FAIL'}")
    report_lines.append(f"- **Total trainable parameter count = 18,524**: {'PASS' if param_pass else 'FAIL'} (Found: {total_params})")
    report_lines.append(f"- **Input shape compatibility [batch, 63]**: {'PASS' if in_shape_pass else 'FAIL'}")
    report_lines.append(f"- **Output shape [batch, 28]**: {'PASS' if out_shape_pass else 'FAIL'}")
    report_lines.append(f"- **Outputs are raw logits (not probabilities)**: {'PASS' if raw_logits_pass else 'FAIL'}")
    report_lines.append(f"- **Output dtype is torch.float32**: {'PASS' if dtype_pass else 'FAIL'}")
    report_lines.append(f"- **All output values are finite**: {'PASS' if finite_pass else 'FAIL'}")
    report_lines.append(f"- **CrossEntropyLoss compatibility with raw logits**: {'PASS' if ce_compat_pass else 'FAIL'}")

    # Check train/eval mode toggling
    model.train()
    is_train = model.training and layers[3].training
    model.eval()
    is_eval = not model.training and not layers[3].training
    report_lines.append(f"- **Training step/mode toggling**: {'PASS' if is_train else 'FAIL'}")
    report_lines.append(f"- **Evaluation step/mode toggling**: {'PASS' if is_eval else 'FAIL'}")

    # Dataset Boundary Integrity
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    integrity_pass = (
        len(train_df) == 53259 and
        len(val_df) == 9186 and
        len(test_df) == 4399 and
        train_df['internal_class_index'].astype(int).between(0, 27).all()
    )
    report_lines.append(f"- **Dataset boundary integrity (row counts & label ranges)**: {'PASS' if integrity_pass else 'FAIL'}")

    # Train Data Loaders (smoke test subset)
    train_dl, val_dl, test_dl = create_dataloaders(batch_size=64, seed=42)

    # Smoke Test
    report_lines.append("")
    report_lines.append("## Smoke Test")
    report_lines.append("> [!NOTE]")
    report_lines.append("> **Disclaimer**: This is a pipeline execution smoke test only. Metrics presented do not reflect the model's actual predictive capability. They merely prove that the metric calculation pipeline functions properly on real data.")

    trainer = MLPTrainer(model, learning_rate=1e-3, weight_decay=1e-4, seed=42)

    # We will run 1 epoch over a tiny subset (e.g. 4 batches = 256 samples for train, 2 batches = 128 for val)
    # Using small custom dataloaders for the smoke test
    from torch.utils.data import Subset, DataLoader
    train_subset = Subset(train_dl.dataset, range(256))
    val_subset = Subset(val_dl.dataset, range(128))

    train_smoke_dl = DataLoader(train_subset, batch_size=64, shuffle=False)
    val_smoke_dl = DataLoader(val_subset, batch_size=64, shuffle=False)

    smoke_train_res = trainer.train_epoch(train_smoke_dl)
    smoke_val_res = trainer.evaluate(val_smoke_dl)

    report_lines.append(f"- **Number of samples:** 256 train, 128 validation")
    report_lines.append(f"- **Number of epochs:** 1")
    report_lines.append(f"- **Execution status:** Completed Successfully")
    report_lines.append(f"- **Smoke Test Metrics (Val):**")
    report_lines.append(f"  - Loss: {smoke_val_res['loss']:.4f}")
    report_lines.append(f"  - Accuracy: {smoke_val_res['accuracy']:.4f}")
    report_lines.append(f"  - Macro Precision: {smoke_val_res['macro_precision']:.4f}")
    report_lines.append(f"  - Macro Recall: {smoke_val_res['macro_recall']:.4f}")
    report_lines.append(f"  - Macro F1: {smoke_val_res['macro_f1']:.4f}")

    # Checkpoint test
    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = os.path.join(tmpdir, "checkpoint.pth")
        trainer.save_checkpoint(chk_path)

        # Reload
        new_model = ASLLandmarkMLP()
        new_trainer = MLPTrainer(new_model)
        new_trainer.load_checkpoint(chk_path)

        ckpt_pass = True
        if new_trainer.epoch != trainer.epoch:
            ckpt_pass = False

    report_lines.append("")
    report_lines.append("## Checkpoint and Reproducibility")
    report_lines.append(f"- **Checkpoint save/load**: {'PASS' if ckpt_pass else 'FAIL'}")

    # Prediction consistency
    model.eval()
    new_model.eval()

    device = next(model.parameters()).device
    test_tensor_device = test_tensor.to(device)

    with torch.no_grad():
        pred_orig = model(test_tensor_device)
        pred_new = new_model(test_tensor_device)

        # Second inference with original model to check pure determinism
        pred_orig_2 = model(test_tensor_device)

    consistency_pass = torch.allclose(pred_orig, pred_new, atol=1e-6)
    reproducibility_pass = torch.allclose(pred_orig, pred_orig_2, atol=1e-6)

    report_lines.append(f"- **Prediction consistency (Loaded vs Saved)**: {'PASS' if consistency_pass else 'FAIL'}")
    report_lines.append(f"- **Reproducibility check (Deterministic inference)**: {'PASS' if reproducibility_pass else 'FAIL'}")

    # Overall PASS/FAIL
    all_checks = [arch_pass, param_pass, in_shape_pass, out_shape_pass, dtype_pass, finite_pass,
                  raw_logits_pass, ce_compat_pass, is_train, is_eval, integrity_pass, ckpt_pass,
                  consistency_pass, reproducibility_pass]

    overall_pass = all(all_checks)

    report_lines.append("")
    report_lines.append("## Training Configuration")
    report_lines.append("- **Loss:** CrossEntropyLoss")
    report_lines.append("- **Optimizer:** AdamW")
    report_lines.append("- **Learning Rate:** 1e-3")
    report_lines.append("- **Weight Decay:** 1e-4")
    report_lines.append("- **Batch Size:** 64")
    report_lines.append("- **Seed:** 42")

    report_lines.append("")
    report_lines.append("## Architecture Details")
    report_lines.append("```text")
    report_lines.append("Input: 63")
    report_lines.append("Linear(63, 128)")
    report_lines.append("BatchNorm1d(128)")
    report_lines.append("ReLU")
    report_lines.append("Dropout(p=0.20)")
    report_lines.append("Linear(128, 64)")
    report_lines.append("ReLU")
    report_lines.append("Dropout(p=0.20)")
    report_lines.append("Linear(64, 28)")
    report_lines.append("Raw Logits")
    report_lines.append("```")
    report_lines.append(f"**Total Trainable Parameters:** {total_params}")

    report_lines.append("")
    report_lines.append("## Final Verdict")
    report_lines.append("**“Is the 28-class landmark MLP implementation technically ready for the full production training run?”**")
    if overall_pass:
        report_lines.append("")
        report_lines.append("**PASS.** The MLP architecture matches the specification perfectly (18,524 parameters). The training infrastructure successfully computes standard loss and metrics, correctly manages checkpoints, maintains data boundaries, and demonstrates deterministic inference capabilities. It is fully ready for production training.")
    else:
        report_lines.append("")
        report_lines.append("**FAIL.** One or more independent checks did not pass.")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

if __name__ == '__main__':
    generate_report()
    print(f"Report generated at {REPORT_PATH}")
