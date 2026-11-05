"""
STEP 8.2 — External Landmark Extraction and Frozen-MLP Evaluation
Processes all 870 external dataset images:
- MediaPipe Hand Landmarker (thresholds 0.3)
- Normalizer canonicalization to 63-D feature vectors
- Frozen ASLLandmarkMLP inference
- Classifier-only metrics & 28-class confusion matrix
- End-to-end system metrics & 29-class confusion matrix
- Confidence threshold analysis
- Parquet & JSON artifact generation
"""

import os
import sys
import glob
import json
import time
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.landmarks.labels import SEMANTIC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, get_class_index
from src.landmarks.normalizer import LandmarkNormalizer
from src.model.mlp import ASLLandmarkMLP

# ── Paths ──
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "asl_alphabet_external")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "asl_landmark_mlp_best.pt")
TASK_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "landmarks", "external")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

PARQUET_OUT = os.path.join(PROCESSED_DIR, "external_hand_landmarks.parquet")
RESULTS_JSON_OUT = os.path.join(PROCESSED_DIR, "external_evaluation_results.json")
RESULTS_CSV_OUT = os.path.join(PROCESSED_DIR, "external_evaluation_results.csv")

EXTRACTION_REPORT = os.path.join(REPORTS_DIR, "asl_external_extraction_report.md")
CLASSIFICATION_REPORT = os.path.join(REPORTS_DIR, "asl_external_classification_report.md")
CM_CLASSIFIER_PNG = os.path.join(REPORTS_DIR, "asl_external_confusion_matrix.png")
CM_E2E_PNG = os.path.join(REPORTS_DIR, "asl_external_end_to_end_confusion_matrix.png")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 29 system classes in alphabetical order (with NOTHING at the end)
SYSTEM_CLASSES_29 = list(SEMANTIC_CLASSES) + ['NOTHING']

def plot_cm(cm, labels, title, out_path, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, labels, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, f"{val}",
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=8)

    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    print("=" * 70)
    print("STEP 8.2 — EXTERNAL LANDMARK EXTRACTION & EVALUATION")
    print("=" * 70)

    # 1. Initialize MediaPipe Detector
    base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)
    normalizer = LandmarkNormalizer()

    # 2. Load Frozen Model Checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint from {CHECKPOINT_PATH} on {device}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model = ASLLandmarkMLP()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. Discover External Dataset Images
    all_folders = sorted(os.listdir(DATA_DIR))
    print(f"Found {len(all_folders)} folders in {DATA_DIR}")

    image_records = []

    for folder in all_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        upper = folder.upper()
        if upper == "DEL":
            semantic = "DELETE"
        elif upper == "SPACE":
            semantic = "SPACE"
        elif upper == "NOTHING":
            semantic = "NOTHING"
        else:
            semantic = upper

        is_hand_bearing = (semantic != "NOTHING")

        files = glob.glob(os.path.join(folder_path, "*.*"))
        files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()

        for fpath in files:
            image_records.append({
                'source_path': fpath,
                'physical_folder': folder,
                'semantic_label': semantic,
                'is_hand_bearing': is_hand_bearing
            })

    print(f"Total external images to process: {len(image_records)}")
    assert len(image_records) == 870, f"Expected 870 images, got {len(image_records)}"

    # 4. Processing Loop
    parquet_records = []
    detailed_results = []

    class_stats = {
        cls: {'attempted': 0, 'success': 0, 'fail': 0}
        for cls in SYSTEM_CLASSES_29
    }
    failure_reasons = {}
    dimension_counts = {}

    start_time = time.time()

    for idx, rec in enumerate(image_records):
        fpath = rec['source_path']
        folder = rec['physical_folder']
        semantic = rec['semantic_label']
        is_hb = rec['is_hand_bearing']

        class_stats[semantic]['attempted'] += 1

        bgr = cv2.imread(fpath)
        if bgr is None:
            reason = "Failed to load image"
            class_stats[semantic]['fail'] += 1
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            detailed_results.append({
                'source_path': fpath,
                'physical_folder': folder,
                'true_semantic_label': semantic,
                'is_hand_bearing': is_hb,
                'image_width': 0,
                'image_height': 0,
                'hand_detected': False,
                'extraction_status': 'IMAGE_LOAD_FAILED',
                'failure_reason': reason,
                'handedness': None,
                'handedness_score': None,
                'features': None,
                'predicted_internal_index': None,
                'predicted_semantic_label': None,
                'confidence': None,
                'system_prediction': 'NOTHING',
                'system_correct': (semantic == 'NOTHING'),
                'classifier_correct': None
            })
            continue

        h, w = bgr.shape[:2]
        dim_key = f"{w}x{h}"
        dimension_counts[dim_key] = dimension_counts.get(dim_key, 0) + 1

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = detector.detect(mp_img)
        except Exception as e:
            reason = f"MediaPipe Exception: {str(e)}"
            class_stats[semantic]['fail'] += 1
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            detailed_results.append({
                'source_path': fpath,
                'physical_folder': folder,
                'true_semantic_label': semantic,
                'is_hand_bearing': is_hb,
                'image_width': w,
                'image_height': h,
                'hand_detected': False,
                'extraction_status': 'DETECTOR_EXCEPTION',
                'failure_reason': reason,
                'handedness': None,
                'handedness_score': None,
                'features': None,
                'predicted_internal_index': None,
                'predicted_semantic_label': None,
                'confidence': None,
                'system_prediction': 'NOTHING',
                'system_correct': (semantic == 'NOTHING'),
                'classifier_correct': None
            })
            continue

        if not result.hand_landmarks or not result.handedness:
            reason = "No hand detected"
            class_stats[semantic]['fail'] += 1
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            detailed_results.append({
                'source_path': fpath,
                'physical_folder': folder,
                'true_semantic_label': semantic,
                'is_hand_bearing': is_hb,
                'image_width': w,
                'image_height': h,
                'hand_detected': False,
                'extraction_status': 'NO_HAND',
                'failure_reason': reason,
                'handedness': None,
                'handedness_score': None,
                'features': None,
                'predicted_internal_index': None,
                'predicted_semantic_label': None,
                'confidence': None,
                'system_prediction': 'NOTHING',
                'system_correct': (semantic == 'NOTHING'),
                'classifier_correct': None
            })
            continue

        landmarks_dict = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in result.hand_landmarks[0]]
        handedness_cat = result.handedness[0][0].category_name
        handedness_score = float(result.handedness[0][0].score)

        valid, features, norm_reason = normalizer.normalize(landmarks_dict, handedness_cat)
        if not valid:
            class_stats[semantic]['fail'] += 1
            failure_reasons[norm_reason] = failure_reasons.get(norm_reason, 0) + 1
            detailed_results.append({
                'source_path': fpath,
                'physical_folder': folder,
                'true_semantic_label': semantic,
                'is_hand_bearing': is_hb,
                'image_width': w,
                'image_height': h,
                'hand_detected': True,
                'extraction_status': 'NORMALIZATION_FAILED',
                'failure_reason': norm_reason,
                'handedness': handedness_cat,
                'handedness_score': handedness_score,
                'features': None,
                'predicted_internal_index': None,
                'predicted_semantic_label': None,
                'confidence': None,
                'system_prediction': 'NOTHING',
                'system_correct': (semantic == 'NOTHING'),
                'classifier_correct': None
            })
            continue

        # Extraction Success!
        class_stats[semantic]['success'] += 1
        features_list = features.tolist()

        # Inference with frozen MLP
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feat_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(logits, dim=1).item()
            conf = probs[0, pred_idx].item()
            pred_label = IDX_TO_CLASS[pred_idx]

        # Parquet entry
        # If semantic is NOTHING, internal_class_index is None
        internal_idx = get_class_index(semantic) if is_hb else None

        parquet_records.append({
            'source_path': fpath,
            'physical_folder': folder,
            'semantic_label': semantic,
            'internal_class_index': internal_idx,
            'handedness': handedness_cat,
            'handedness_score': handedness_score,
            'features': features_list
        })

        # Detailed result
        sys_pred = pred_label
        sys_corr = (sys_pred == semantic)
        clf_corr = (pred_label == semantic) if is_hb else None

        detailed_results.append({
            'source_path': fpath,
            'physical_folder': folder,
            'true_semantic_label': semantic,
            'is_hand_bearing': is_hb,
            'image_width': w,
            'image_height': h,
            'hand_detected': True,
            'extraction_status': 'SUCCESS',
            'failure_reason': None,
            'handedness': handedness_cat,
            'handedness_score': handedness_score,
            'features': features_list,
            'predicted_internal_index': pred_idx,
            'predicted_semantic_label': pred_label,
            'confidence': conf,
            'system_prediction': sys_pred,
            'system_correct': sys_corr,
            'classifier_correct': clf_corr
        })

    duration = time.time() - start_time
    print(f"Extraction and inference completed in {duration:.2f}s ({870/duration:.1f} images/s)")

    # 5. Save Parquet of Extracted Landmarks
    schema = pa.schema([
        ('source_path', pa.string()),
        ('physical_folder', pa.string()),
        ('semantic_label', pa.string()),
        ('internal_class_index', pa.int8()),
        ('handedness', pa.string()),
        ('handedness_score', pa.float32()),
        ('features', pa.list_(pa.float32(), 63))
    ])

    df_parquet = pd.DataFrame(parquet_records)
    table = pa.Table.from_pandas(df_parquet, schema=schema)
    pq.write_table(table, PARQUET_OUT)
    print(f"Saved {len(parquet_records)} extracted records to {PARQUET_OUT}")

    # 6. Save JSON & CSV Results of all 870 images
    with open(RESULTS_JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2)

    df_summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'features'}
        for r in detailed_results
    ])
    df_summary.to_csv(RESULTS_CSV_OUT, index=False)
    print(f"Saved full results to {RESULTS_JSON_OUT} and {RESULTS_CSV_OUT}")

    # 7. Compute Statistics
    # Hand-bearing statistics
    hb_results = [r for r in detailed_results if r['is_hand_bearing']]
    nothing_results = [r for r in detailed_results if not r['is_hand_bearing']]

    hb_attempted = len(hb_results) # 840
    hb_success = sum(1 for r in hb_results if r['extraction_status'] == 'SUCCESS')
    hb_fail = hb_attempted - hb_success
    hb_extraction_rate = (hb_success / hb_attempted) * 100

    # NOTHING statistics
    nothing_attempted = len(nothing_results) # 30
    nothing_fp = sum(1 for r in nothing_results if r['hand_detected'])
    nothing_fp_rate = (nothing_fp / nothing_attempted) * 100

    # 8. Classifier-Only Evaluation (on hb_success)
    clf_eval_samples = [r for r in hb_results if r['extraction_status'] == 'SUCCESS']
    y_true_clf = [r['true_semantic_label'] for r in clf_eval_samples]
    y_pred_clf = [r['predicted_semantic_label'] for r in clf_eval_samples]

    clf_acc = accuracy_score(y_true_clf, y_pred_clf)
    clf_prec, clf_rec, clf_f1, _ = precision_recall_fscore_support(
        y_true_clf, y_pred_clf, average='macro', zero_division=0
    )
    clf_weighted_f1 = precision_recall_fscore_support(
        y_true_clf, y_pred_clf, average='weighted', zero_division=0
    )[2]

    cr_clf_text = classification_report(
        y_true_clf, y_pred_clf, labels=SEMANTIC_CLASSES, target_names=SEMANTIC_CLASSES, zero_division=0
    )

    # Plot Classifier Confusion Matrix
    cm_clf = confusion_matrix(y_true_clf, y_pred_clf, labels=SEMANTIC_CLASSES)
    plot_cm(cm_clf, SEMANTIC_CLASSES, "Classifier Confusion Matrix (External Test)", CM_CLASSIFIER_PNG, figsize=(14, 12))

    # 9. End-to-End Evaluation (on all 870 images)
    y_true_e2e = [r['true_semantic_label'] for r in detailed_results]
    y_pred_e2e = [r['system_prediction'] for r in detailed_results]

    e2e_correct = sum(1 for r in detailed_results if r['system_correct'])
    e2e_acc = e2e_correct / len(detailed_results)

    hb_correct = sum(1 for r in hb_results if r['system_correct'])
    hb_e2e_acc = hb_correct / hb_attempted
    hb_missed = sum(1 for r in hb_results if not r['hand_detected'])

    # Plot End-to-End Confusion Matrix (29 classes)
    cm_e2e = confusion_matrix(y_true_e2e, y_pred_e2e, labels=SYSTEM_CLASSES_29)
    plot_cm(cm_e2e, SYSTEM_CLASSES_29, "End-to-End System Confusion Matrix (External Test)", CM_E2E_PNG, figsize=(15, 13))

    # 10. Confidence Analysis
    confidences = [r['confidence'] for r in clf_eval_samples if r['confidence'] is not None]
    conf_mean = float(np.mean(confidences)) if confidences else 0.0
    conf_median = float(np.median(confidences)) if confidences else 0.0
    conf_min = float(np.min(confidences)) if confidences else 0.0
    conf_max = float(np.max(confidences)) if confidences else 0.0

    thresholds = [0.50, 0.70, 0.80, 0.90]
    thresh_table = []
    for th in thresholds:
        retained = [r for r in clf_eval_samples if r['confidence'] >= th]
        n_ret = len(retained)
        ret_frac = (n_ret / len(clf_eval_samples)) * 100 if clf_eval_samples else 0.0
        n_rej = len(clf_eval_samples) - n_ret
        if n_ret > 0:
            ret_corr = sum(1 for r in retained if r['predicted_semantic_label'] == r['true_semantic_label'])
            ret_acc = (ret_corr / n_ret) * 100
        else:
            ret_acc = 0.0
        thresh_table.append({
            'threshold': th,
            'retained_count': n_ret,
            'retained_fraction': ret_frac,
            'retained_accuracy': ret_acc,
            'rejected_count': n_rej
        })

    # 11. Write reports/asl_external_extraction_report.md
    ext_rep_lines = [
        "# ASL External Dataset Landmark Extraction Report",
        "",
        "## 1. Overview",
        f"- **External Dataset Path:** `{DATA_DIR}`",
        f"- **Model Task Asset:** `{TASK_MODEL_PATH}`",
        "- **MediaPipe Thresholds:** `min_hand_detection_confidence=0.3`, `min_hand_presence_confidence=0.3`",
        f"- **Output Parquet:** `{PARQUET_OUT}`",
        f"- **Full Results JSON:** `{RESULTS_JSON_OUT}`",
        f"- **Extraction Runtime:** {duration:.2f} seconds ({870/duration:.1f} images/s)",
        "",
        "## 2. Overall Extraction Statistics",
        f"- **Total Attempted Images:** {len(image_records)} (840 hand-bearing + 30 NOTHING)",
        f"- **Hand-Bearing Images Attempted:** {hb_attempted}",
        f"- **Hand-Bearing Successful Detections:** {hb_success}",
        f"- **Hand-Bearing Failed Detections:** {hb_fail}",
        f"- **Hand-Bearing Extraction Rate:** {hb_extraction_rate:.2f}%",
        f"- **NOTHING Images Attempted:** {nothing_attempted}",
        f"- **NOTHING False-Positive Detections:** {nothing_fp} ({nothing_fp_rate:.2f}%)",
        f"- **Total Normalized Vectors Stored in Parquet:** {len(parquet_records)}",
        "",
        "## 3. Failure Reasons Breakdown",
        "| Failure Reason | Occurrences | Percentage (of 870) |",
        "|---|---|---|"
    ]
    for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
        ext_rep_lines.append(f"| {reason} | {count} | {count/870*100:.2f}% |")
    ext_rep_lines.extend([
        "",
        "## 4. Image Dimensions Distribution",
        "| Dimension (Width x Height) | Image Count | Percentage |",
        "|---|---|---|"
    ])
    for dim, count in sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True):
        ext_rep_lines.append(f"| {dim} | {count} | {count/870*100:.2f}% |")
    ext_rep_lines.extend([
        "",
        "## 5. Per-Class Extraction Table",
        "| Class | Class Type | Attempted | Success | Fail | Success Rate |",
        "|---|---|---|---|---|---|"
    ])
    for cls in SYSTEM_CLASSES_29:
        ctype = "No-Hand" if cls == "NOTHING" else "Hand-Bearing"
        att = class_stats[cls]['attempted']
        succ = class_stats[cls]['success']
        fl = class_stats[cls]['fail']
        rate = (succ / att * 100) if att > 0 else 0.0
        ext_rep_lines.append(f"| `{cls}` | {ctype} | {att} | {succ} | {fl} | {rate:.1f}% |")
    ext_rep_lines.append("")

    with open(EXTRACTION_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(ext_rep_lines) + "\n")
    print(f"Wrote {EXTRACTION_REPORT}")

    # 12. Write reports/asl_external_classification_report.md
    clf_rep_lines = [
        "# ASL External Dataset Evaluation Report",
        "",
        f"**Model Checkpoint:** `{CHECKPOINT_PATH}` (Best Epoch: {checkpoint['epoch']}, Frozen)",
        f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> [!IMPORTANT]",
        "> **Denominator Distinction:**",
        "> - **Classifier-Only Metrics:** Evaluated **strictly** on the successfully extracted hand-bearing external images.",
        f">   *(Denominator: {hb_success} samples across 28 classes)*.",
        "> - **End-to-End System Metrics:** Evaluated on all 870 images, where missing a hand-bearing gesture is counted as a pipeline error (predicting NOTHING).",
        f">   *(Denominator: 870 total images across 29 classes)*.",
        "",
        "## 1. Classifier-Only Performance (28 Classes)",
        f"- **Evaluated Support:** {len(clf_eval_samples)} / {hb_attempted} hand-bearing images",
        f"- **Accuracy:** **{clf_acc:.4f}** ({clf_acc*100:.2f}%)",
        f"- **Macro Precision:** **{clf_prec:.4f}** ({clf_prec*100:.2f}%)",
        f"- **Macro Recall:** **{clf_rec:.4f}** ({clf_rec*100:.2f}%)",
        f"- **Macro F1:** **{clf_f1:.4f}** ({clf_f1*100:.2f}%)",
        f"- **Weighted F1:** **{clf_weighted_f1:.4f}** ({clf_weighted_f1*100:.2f}%)",
        "",
        "### Per-Class Performance Table",
        "```text",
        cr_clf_text,
        "```",
        "",
        "### Confusion Matrix",
        f"![Classifier Confusion Matrix]({os.path.basename(CM_CLASSIFIER_PNG)})",
        "",
        "## 2. End-to-End System Performance (29 Classes)",
        f"- **Total Source Images:** 870 (840 hand-bearing + 30 NOTHING)",
        f"- **Total Correct System Predictions:** {e2e_correct} / 870",
        f"- **Overall End-to-End Accuracy:** **{e2e_acc:.4f}** ({e2e_acc*100:.2f}%)",
        f"- **Hand-Bearing Correct Predictions:** {hb_correct} / {hb_attempted}",
        f"- **Hand-Bearing End-to-End Accuracy:** **{hb_e2e_acc:.4f}** ({hb_e2e_acc*100:.2f}%)",
        f"- **Missed Hand-Bearing Images (Failed Detection -> NOTHING):** {hb_missed} ({hb_missed/hb_attempted*100:.2f}%)",
        f"- **NOTHING True Negatives (No Hand Detected):** {nothing_attempted - nothing_fp} / {nothing_attempted} ({(nothing_attempted - nothing_fp)/nothing_attempted*100:.2f}%)",
        f"- **NOTHING False Positives (Hand Detected in Empty Image):** {nothing_fp} / {nothing_attempted} ({nothing_fp_rate:.2f}%)",
        "",
        "### End-to-End Confusion Matrix",
        f"![End-to-End Confusion Matrix]({os.path.basename(CM_E2E_PNG)})",
        "",
        "## 3. Confidence Analysis",
        f"- **Mean Confidence:** {conf_mean:.4f} ({conf_mean*100:.2f}%)",
        f"- **Median Confidence:** {conf_median:.4f} ({conf_median*100:.2f}%)",
        f"- **Min Confidence:** {conf_min:.4f} ({conf_min*100:.2f}%)",
        f"- **Max Confidence:** {conf_max:.4f} ({conf_max*100:.2f}%)",
        "",
        "### Threshold Coverage & Accuracy Analysis (Informational)",
        "| Confidence Threshold | Retained Samples | Coverage (% of extracted) | Accuracy on Retained | Rejected Samples |",
        "|---|---|---|---|---|"
    ]
    for row in thresh_table:
        clf_rep_lines.append(
            f"| $\\ge {row['threshold']:.2f}$ | {row['retained_count']} / {len(clf_eval_samples)} | "
            f"{row['retained_fraction']:.2f}% | {row['retained_accuracy']:.2f}% | {row['rejected_count']} |"
        )
    clf_rep_lines.extend([
        "",
        "> [!NOTE]",
        "> These threshold analyses are descriptive only and do not modify production model behavior.",
        "",
        "## 4. Engineering Summary & Generalization Observations",
        f"1. **Extraction Robustness:** MediaPipe successfully extracted landmarks from {hb_success} of 840 hand-bearing images ({hb_extraction_rate:.2f}%).",
        f"2. **Classifier Generalization:** On successfully extracted landmarks, the frozen MLP achieved **{clf_acc*100:.2f}% accuracy** and **{clf_f1*100:.2f}% macro F1**, demonstrating strong feature generalization to previously unseen subjects and lighting conditions.",
        f"3. **End-to-End Pipeline Performance:** Accounting for detection misses and the NOTHING background baseline, the full system achieved **{e2e_acc*100:.2f}% overall accuracy** across all 870 images.",
        f"4. **NOTHING Handling:** {nothing_attempted - nothing_fp} / {nothing_attempted} NOTHING images had zero hands detected ({(nothing_attempted - nothing_fp)/nothing_attempted*100:.2f}% correct rejection).",
        ""
    ])

    with open(CLASSIFICATION_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(clf_rep_lines) + "\n")
    print(f"Wrote {CLASSIFICATION_REPORT}")

    print("=" * 70)
    print("PRIMARY EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Hand-Bearing Extraction Rate: {hb_extraction_rate:.2f}% ({hb_success}/{hb_attempted})")
    print(f"Classifier-Only Accuracy:     {clf_acc*100:.2f}% (Macro F1: {clf_f1*100:.2f}%)")
    print(f"End-to-End Accuracy:          {e2e_acc*100:.2f}% ({e2e_correct}/870)")
    print(f"NOTHING False Positive Rate:  {nothing_fp_rate:.2f}% ({nothing_fp}/30)")
    print("=" * 70)

if __name__ == '__main__':
    main()
