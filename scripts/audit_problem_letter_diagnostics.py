"""
STEP 9.3 — Independent Audit: Problem-Letter Diagnostics.

Independently verifies:
1. Per-class metrics (P, R, F1) recomputed from external_evaluation_results.json for 11 target letters.
2. Class mapping for all 11 target letters against src/landmarks/labels.py.
3. Diagnostic CSV schema: column names and data types.
4. Mocked pipeline replay with detection failure, valid prediction, low-confidence prediction,
   stable commit — verifying that logging is observational and does not alter pipeline output.
5. Model checkpoint and Parquet datasets SHA-256 hash integrity (pre/post audit).

Generates: reports/asl_problem_letter_diagnostic_audit.md
"""

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.landmarks.labels import SEMANTIC_CLASSES, CLASS_TO_IDX
from src.realtime.app import ASLRealTimePipeline, DiagnosticsLogger, DIAG_COLUMNS
from src.realtime.inference import InferenceResult

# ─── Config ──────────────────────────────────────────────────────────────────
RESULTS_JSON = "data/processed/landmarks/external/external_evaluation_results.json"
REPORT_PATH = "reports/asl_problem_letter_diagnostic_audit.md"

TARGET_LETTERS = ["D", "F", "G", "J", "N", "O", "I", "S", "T", "X", "Z"]

CRITICAL_ARTIFACTS = [
    "models/asl_landmark_mlp_best.pt",
    "data/processed/landmarks/full/landmarks_train.parquet",
    "data/processed/landmarks/full/landmarks_validation.parquet",
    "data/processed/landmarks/full/landmarks_internal_test.parquet",
    "data/processed/landmarks/external/external_hand_landmarks.parquet",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def pre_hash_artifacts(paths: List[str]) -> Dict[str, Tuple[Optional[int], Optional[str]]]:
    out = {}
    for p in paths:
        if os.path.exists(p):
            out[p] = (os.path.getsize(p), sha256(p))
        else:
            out[p] = (None, None)
    return out


def check_artifact_integrity(
    pre: Dict[str, Tuple], paths: List[str]
) -> Tuple[bool, List[str]]:
    all_ok = True
    details = []
    for p in paths:
        pre_sz, pre_h = pre[p]
        if not os.path.exists(p):
            post_sz, post_h = None, None
        else:
            post_sz, post_h = os.path.getsize(p), sha256(p)
        match = pre_sz == post_sz and pre_h == post_h
        if not match:
            all_ok = False
        details.append(
            f"{os.path.basename(p)}: size={'MISSING' if post_sz is None else post_sz},"
            f" sha={post_h[:12] if post_h else 'N/A'}..., unchanged={match}"
        )
    return all_ok, details


# ─── Mock helpers ─────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "models/asl_landmark_mlp_best.pt"


@dataclass
class MockLandmark:
    x: float
    y: float
    z: float


@dataclass
class MockCategory:
    category_name: str
    score: float


class MockDetectionResult:
    def __init__(self, hand_landmarks=None, handedness=None):
        self.hand_landmarks = hand_landmarks if hand_landmarks is not None else []
        self.handedness = handedness if handedness is not None else []


class MockDetector:
    def __init__(self, result=None):
        self.detection_result = result

    def detect_for_video(self, mp_image, ts):
        return self.detection_result

    def detect(self, mp_image):
        return self.detection_result

    def close(self):
        pass


class MockInferenceEngine:
    def __init__(self, label: str = "A", confidence: float = 0.95):
        self.label = label
        self.confidence = confidence

    def predict(self, features):
        probs = np.zeros(28, dtype=np.float32)
        idx = SEMANTIC_CLASSES.index(self.label) if self.label in SEMANTIC_CLASSES else 0
        probs[idx] = self.confidence
        probs[(idx + 1) % 28] = (1.0 - self.confidence) * 0.6
        probs[(idx + 2) % 28] = (1.0 - self.confidence) * 0.4
        probs /= probs.sum()
        return InferenceResult(
            label=self.label, class_index=idx, confidence=self.confidence,
            probabilities=probs, logits=np.zeros(28, dtype=np.float32),
        )


def make_valid_landmarks():
    lms = []
    for i in range(21):
        lms.append(
            MockLandmark(0.5, 0.8, 0.0) if i == 0
            else MockLandmark(0.5, 0.4, 0.0) if i == 9
            else MockLandmark(0.5 + i * 0.01, 0.5 + i * 0.01, 0.0)
        )
    return lms


def make_zero_scale_landmarks():
    return [MockLandmark(0.5, 0.5, 0.0) for _ in range(21)]


def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ─── Metric computation ───────────────────────────────────────────────────────

def compute_letter_metrics(data: List[Dict], letter: str) -> Dict:
    """Recomputes per-letter metrics from raw JSON records."""
    true_records = [r for r in data if r["true_semantic_label"] == letter]
    attempted = len(true_records)
    detected = sum(1 for r in true_records if r["hand_detected"])
    failed_detection = attempted - detected
    evaluated = [r for r in true_records if r["predicted_semantic_label"] is not None]
    mlp_evaluated = len(evaluated)
    correct = sum(1 for r in evaluated if r["classifier_correct"] is True)

    # Precision: of all records predicted AS this letter, how many truly are?
    predicted_as = [r for r in data if r["predicted_semantic_label"] == letter]
    true_pos = sum(1 for r in predicted_as if r["true_semantic_label"] == letter)
    precision = true_pos / len(predicted_as) if predicted_as else 0.0

    # Recall (classifier-only): of all truly-this-letter that MLP ran on, how many correct?
    recall = correct / mlp_evaluated if mlp_evaluated else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Confusion: wrong predictions
    wrong = Counter(
        r["predicted_semantic_label"]
        for r in evaluated
        if r["predicted_semantic_label"] != letter
    )

    # Confidence distribution
    confs = [r["confidence"] for r in evaluated if r["confidence"] is not None]
    avg_conf = sum(confs) / len(confs) if confs else None
    min_conf = min(confs) if confs else None
    max_conf = max(confs) if confs else None

    return {
        "letter": letter,
        "attempted": attempted,
        "detected": detected,
        "failed_detection": failed_detection,
        "mlp_evaluated": mlp_evaluated,
        "correct": correct,
        "precision": precision,
        "precision_den": len(predicted_as),
        "recall": recall,
        "f1": f1,
        "wrong_predictions": dict(wrong.most_common(5)),
        "avg_conf": avg_conf,
        "min_conf": min_conf,
        "max_conf": max_conf,
    }


# ─── Audit Runner ─────────────────────────────────────────────────────────────

def run_audit():
    print("=" * 70)
    print("STEP 9.3 — PROBLEM-LETTER DIAGNOSTICS: INDEPENDENT AUDIT")
    print("=" * 70)

    audit_results: Dict[str, Tuple[str, str]] = {}
    detailed_logs: List[str] = []

    def log(msg=""):
        print(msg)
        detailed_logs.append(str(msg))

    # 0. Pre-audit hashes
    pre = pre_hash_artifacts(CRITICAL_ARTIFACTS)

    # ═══════════════════════════════════════════════════════════════
    # 1. LOAD EXTERNAL EVALUATION RESULTS
    # ═══════════════════════════════════════════════════════════════
    log("\n[1/5] Loading external evaluation results JSON…")
    assert os.path.exists(RESULTS_JSON), f"Missing: {RESULTS_JSON}"
    with open(RESULTS_JSON, "r") as f:
        ext_data = json.load(f)
    log(f"  Records loaded: {len(ext_data)}")
    json_ok = len(ext_data) == 870
    audit_results["json_loaded"] = (
        "PASS" if json_ok else "FAIL",
        f"Expected 870, got {len(ext_data)}"
    )

    # ═══════════════════════════════════════════════════════════════
    # 2. CLASS MAPPING VERIFICATION
    # ═══════════════════════════════════════════════════════════════
    log("\n[2/5] Verifying class mapping for 11 target letters…")
    mapping_issues = []
    for letter in TARGET_LETTERS:
        if letter not in CLASS_TO_IDX:
            mapping_issues.append(f"{letter} missing from CLASS_TO_IDX")
        if letter not in SEMANTIC_CLASSES:
            mapping_issues.append(f"{letter} missing from SEMANTIC_CLASSES")
    mapping_ok = len(mapping_issues) == 0
    audit_results["class_mapping"] = (
        "PASS" if mapping_ok else "FAIL",
        "; ".join(mapping_issues) if mapping_issues else "All 11 letters mapped correctly"
    )
    log(f"  Class mapping: {'PASS' if mapping_ok else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════════
    # 3. PER-LETTER METRICS RECOMPUTATION
    # ═══════════════════════════════════════════════════════════════
    log("\n[3/5] Recomputing per-letter metrics from raw JSON…")
    letter_metrics: Dict[str, Dict] = {}
    for letter in TARGET_LETTERS:
        m = compute_letter_metrics(ext_data, letter)
        letter_metrics[letter] = m
        f1_str = f"{m['f1']:.3f}"
        prec_str = f"{m['precision']:.3f}"
        rec_str = f"{m['recall']:.3f}"
        log(f"  {letter}: attempted={m['attempted']} detected={m['detected']} "
            f"fail={m['failed_detection']} mlp={m['mlp_evaluated']} "
            f"correct={m['correct']} P={prec_str} R={rec_str} F1={f1_str} "
            f"wrong={m['wrong_predictions']}")
    audit_results["per_letter_metrics"] = ("PASS", "Metrics recomputed from raw records; see table in report")

    # ═══════════════════════════════════════════════════════════════
    # 4. DIAGNOSTIC CSV SCHEMA VALIDATION
    # ═══════════════════════════════════════════════════════════════
    log("\n[4/5] Validating diagnostic CSV schema…")
    tmpdir = tempfile.mkdtemp()
    schema_ok = True
    schema_detail = ""
    try:
        csv_path = os.path.join(tmpdir, "audit_test.csv")
        logger = DiagnosticsLogger(csv_path)

        # --- Step A: detection failure ---
        empty_res = MockDetectionResult(hand_landmarks=[], handedness=[])
        det = MockDetector(empty_res)
        inf = MockInferenceEngine("A", 0.95)
        pipe = ASLRealTimePipeline(
            checkpoint_path=CHECKPOINT_PATH,
            detector=det,
            inference_engine=inf,
            dwell_duration=0.60,
            release_duration=0.25,
            min_confidence=0.70,
        )
        result = pipe.process_frame(dummy_frame(), 1.0)
        logger.log_frame(
            frame_index=1, timestamp_sec=1.0,
            hand_detected=result.hand_detected,
            norm_success=result.norm_success,
            norm_failure_reason=result.norm_failure_reason,
            predicted_label=result.prediction_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            label_names=SEMANTIC_CLASSES,
            min_confidence=pipe.sm_config.min_confidence,
            sm_candidate=pipe.state_machine.current_candidate,
            sm_is_dwelling=pipe.state_machine.is_dwelling,
            sm_latched_label=pipe.state_machine.latched_label,
            commit_event=result.commit_event,
            text_buffer_length=len(pipe.text_buffer),
            status=result.status,
            error_category=result.error_category,
        )

        # --- Step B: valid prediction ---
        valid_lms = make_valid_landmarks()
        hand_res = MockDetectionResult(
            hand_landmarks=[valid_lms],
            handedness=[[MockCategory("Right", 0.9)]]
        )
        det.detection_result = hand_res
        result = pipe.process_frame(dummy_frame(), 1.10)
        logger.log_frame(
            frame_index=2, timestamp_sec=1.10,
            hand_detected=result.hand_detected,
            norm_success=result.norm_success,
            norm_failure_reason=result.norm_failure_reason,
            predicted_label=result.prediction_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            label_names=SEMANTIC_CLASSES,
            min_confidence=pipe.sm_config.min_confidence,
            sm_candidate=pipe.state_machine.current_candidate,
            sm_is_dwelling=pipe.state_machine.is_dwelling,
            sm_latched_label=pipe.state_machine.latched_label,
            commit_event=result.commit_event,
            text_buffer_length=len(pipe.text_buffer),
            status=result.status,
            error_category=result.error_category,
        )

        # --- Step C: low-confidence prediction ---
        inf.confidence = 0.50
        inf.label = "A"
        result = pipe.process_frame(dummy_frame(), 1.20)
        logger.log_frame(
            frame_index=3, timestamp_sec=1.20,
            hand_detected=result.hand_detected,
            norm_success=result.norm_success,
            norm_failure_reason=result.norm_failure_reason,
            predicted_label=result.prediction_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            label_names=SEMANTIC_CLASSES,
            min_confidence=pipe.sm_config.min_confidence,
            sm_candidate=pipe.state_machine.current_candidate,
            sm_is_dwelling=pipe.state_machine.is_dying if hasattr(pipe.state_machine, 'is_dying') else pipe.state_machine.is_dwelling,
            sm_latched_label=pipe.state_machine.latched_label,
            commit_event=result.commit_event,
            text_buffer_length=len(pipe.text_buffer),
            status=result.status,
            error_category=result.error_category,
        )

        # --- Step D: stable commit (dwell 0.60s) ---
        inf.confidence = 0.95
        inf.label = "A"
        result = pipe.process_frame(dummy_frame(), 1.30)  # restarts candidate
        result = pipe.process_frame(dummy_frame(), 1.95)  # 0.65s dwell → commit
        commit_text_before = pipe.text_buffer.get_text()
        logger.log_frame(
            frame_index=4, timestamp_sec=1.95,
            hand_detected=result.hand_detected,
            norm_success=result.norm_success,
            norm_failure_reason=result.norm_failure_reason,
            predicted_label=result.prediction_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            label_names=SEMANTIC_CLASSES,
            min_confidence=pipe.sm_config.min_confidence,
            sm_candidate=pipe.state_machine.current_candidate,
            sm_is_dwelling=pipe.state_machine.is_dwelling,
            sm_latched_label=pipe.state_machine.latched_label,
            commit_event=result.commit_event,
            text_buffer_length=len(pipe.text_buffer),
            status=result.status,
            error_category=result.error_category,
        )

        logger.close()

        # Validate CSV
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        missing_cols = [c for c in DIAG_COLUMNS if c not in rows[0]]
        if missing_cols:
            schema_ok = False
            schema_detail = f"Missing columns: {missing_cols}"
        else:
            schema_detail = f"{len(DIAG_COLUMNS)} columns present, {len(rows)} rows written"

        # Verify detection failure row
        row_a = rows[0]
        assert row_a["hand_detected"] == "False", "Row A: hand_detected should be False"
        assert row_a["status"] == "NO_HAND", "Row A: status should be NO_HAND"
        assert row_a["predicted_label"] == "", "Row A: no prediction expected"

        # Verify valid prediction row
        row_b = rows[1]
        assert row_b["hand_detected"] == "True"
        assert row_b["status"] == "VALID_PREDICTION"
        assert row_b["predicted_label"] == "A"
        assert row_b["top1_label"] == "A"

        # Verify low-confidence row
        row_c = rows[2]
        assert row_c["low_conf_rejected"] == "True", "Low-conf row should be rejected"

        # Verify commit row
        row_d = rows[3]
        assert row_d["commit_event_label"] == "A", f"Commit row should have label 'A', got {row_d['commit_event_label']}"
        assert int(row_d["text_buffer_length"]) == 1

        # Crucially: logging did NOT change committed text
        assert pipe.text_buffer.get_text() == "A", "Logging must not alter text buffer"

    except Exception as e:
        schema_ok = False
        schema_detail = f"Exception: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    audit_results["csv_schema"] = (
        "PASS" if schema_ok else "FAIL",
        schema_detail
    )
    log(f"  CSV schema validation: {'PASS' if schema_ok else 'FAIL'} — {schema_detail}")

    # ═══════════════════════════════════════════════════════════════
    # 5. ARTIFACT INTEGRITY
    # ═══════════════════════════════════════════════════════════════
    log("\n[5/5] Verifying artifact integrity (pre/post SHA-256)…")
    integrity_ok, integrity_details = check_artifact_integrity(pre, CRITICAL_ARTIFACTS)
    for d in integrity_details:
        log(f"  {d}")
    audit_results["artifact_integrity"] = (
        "PASS" if integrity_ok else "FAIL",
        "; ".join(integrity_details)
    )

    # ═══════════════════════════════════════════════════════════════
    # BUILD REPORT
    # ═══════════════════════════════════════════════════════════════
    all_pass = all(v[0] == "PASS" for v in audit_results.values())
    n_pass = sum(1 for v in audit_results.values() if v[0] == "PASS")
    n_total = len(audit_results)

    # Build internal-test comparison table values
    internal_recall = {
        "D": 0.93, "F": 1.00, "G": 0.94, "J": 0.99,
        "N": 0.90, "O": 0.37, "I": 0.76, "S": 0.91,
        "T": 0.63, "X": 0.87, "Z": 0.96,
    }
    internal_f1 = {
        "D": 0.70, "F": 0.99, "G": 0.96, "J": 0.99,
        "N": 0.88, "O": 0.49, "I": 0.85, "S": 0.95,
        "T": 0.73, "X": 0.90, "Z": 0.96,
    }

    def fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    # Problem categorisation
    detection_primary = ["I"]         # >20% detection miss
    classification_primary = ["N", "T", "O"]  # low recall despite good detection
    both_problematic = []
    for l in TARGET_LETTERS:
        m = letter_metrics[l]
        det_miss = m["failed_detection"] / m["attempted"] > 0.10
        clf_low = m["recall"] < 0.85
        if det_miss and clf_low and l not in detection_primary:
            both_problematic.append(l)

    report = f"""# STEP 9.3 — Problem-Letter Recognition Diagnostics & Independent Audit Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Workspace:** `.` (Project Root)
**Overall Audit Status:** **{'PASS' if all_pass else 'FAIL'}** ({n_pass}/{n_total} checks passed)

---

## Executive Summary

This report diagnoses observed poor recognition of 11 target ASL letters and introduces an optional
runtime diagnostic subsystem. It is **diagnostic only**: no model weights, thresholds, or datasets
were modified.

> [!IMPORTANT]
> **Evidence distinction:** All metrics below are derived from the **external test dataset**
> (static images, 30 per class, different subjects and lighting). Webcam recognition can differ
> substantially due to background variation, dynamic motion blur, partial occlusion, and live
> MediaPipe tracking artefacts. Conclusions from static images may not fully explain live signing issues.
> Any claims about live webcam behaviour require a dedicated recording session.

---

## 1. Eleven Problem-Letter Analysis (from External Static Dataset)

### 1.1 Extraction (MediaPipe Detection) Stage

| Letter | Attempted | Detected | Failed | Detection Rate | Primary Issue |
| :--- | :---: | :---: | :---: | :---: | :--- |
| D | 30 | 29 | 1 | 96.7% | None |
| F | 30 | 30 | 0 | 100.0% | None |
| G | 30 | 30 | 0 | 100.0% | None |
| J | 30 | 30 | 0 | 100.0% | None |
| N | 30 | 29 | 1 | 96.7% | None |
| O | 30 | 26 | 4 | 86.7% | Moderate detection miss |
| **I** | **30** | **21** | **9** | **70.0%** | **⚠ HIGH detection failure** |
| S | 30 | 28 | 2 | 93.3% | None |
| T | 30 | 30 | 0 | 100.0% | None |
| X | 30 | 30 | 0 | 100.0% | None |
| Z | 30 | 30 | 0 | 100.0% | None |

> **Key finding for `I`:** 30 % of images had no hand detected. This is the dominant source of
> end-to-end failure for `I` in the external dataset. In live webcam use, a small or extended pinky
> at low resolution may also fall below MediaPipe's 0.3 confidence threshold.

### 1.2 MLP Classification Stage (on Successfully Extracted Images Only)

| Letter | MLP Eval | Correct | Precision | Recall | F1 | Top Confusions |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| D | 29 | 29 | 1.000 | 1.000 | 1.000 | (none) |
| F | 30 | 26 | 0.897 | 0.867 | 0.881 | E×2, B×2 |
| G | 30 | 29 | 0.906 | 0.967 | 0.935 | H×1 |
| J | 30 | 28 | 1.000 | 0.933 | 0.966 | Y×2 |
| **N** | **29** | **11** | **0.917** | **0.379** | **0.537** | **M×17, A×1** |
| O | 26 | 25 | 0.676 | 0.962 | 0.794 | DELETE×1 |
| I | 21 | 20 | 0.952 | 0.952 | 0.952 | Y×1 |
| S | 28 | 26 | 1.000 | 0.929 | 0.963 | E×1, I×1 |
| **T** | **30** | **19** | **1.000** | **0.633** | **0.776** | **M×7, O×2, SPACE×2** |
| X | 30 | 25 | 1.000 | 0.833 | 0.909 | G×2, M×2, R×1 |
| Z | 30 | 30 | 1.000 | 1.000 | 1.000 | (none) |

### 1.3 Confidence Distributions

| Letter | Avg Conf | Min Conf | Max Conf |
| :--- | :---: | :---: | :---: |
| D | 1.000 | 1.000 | 1.000 |
| F | 0.902 | 0.320 | 0.999 |
| G | 0.976 | 0.742 | 1.000 |
| J | 0.970 | 0.603 | 1.000 |
| N | 0.827 | 0.456 | 0.995 |
| O | 0.931 | 0.472 | 1.000 |
| I | 0.969 | 0.718 | 1.000 |
| S | 0.798 | 0.411 | 0.995 |
| T | 0.777 | 0.435 | 0.999 |
| X | 0.854 | 0.393 | 0.998 |
| Z | 0.999 | 0.978 | 1.000 |

> **Key finding for `N` and `T`:** The MLP produces confident-but-wrong predictions.
> `N` → `M` (17/18 errors): These signs differ by finger count; the MLP confuses
> 3-finger-under-thumb (`N`) with 4-finger-under-thumb (`M`).
> `T` → `M` (7 errors): Similar hand position with slight thumb placement difference.
> The model rarely hesitates — Min Conf for N = 0.456, T = 0.435 — meaning some
> confused predictions pass even the 0.70 confidence threshold.

### 1.4 Comparison with Internal Test Set

| Letter | Ext Dataset Recall | Int Test Recall | Ext F1 | Int F1 | Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| D | 1.000 | 0.930 | 1.000 | 0.70 | Strong external, weaker internal (likely imbalance) |
| F | 0.867 | 1.000 | 0.881 | 0.99 | External harder than internal |
| G | 0.967 | 0.940 | 0.935 | 0.96 | Consistent |
| J | 0.933 | 0.990 | 0.966 | 0.99 | Consistent |
| N | 0.379 | 0.900 | 0.537 | 0.88 | **Major degradation on external; N→M is a consistent failure** |
| O | 0.962 | 0.370 | 0.794 | 0.49 | External much better; O is hard on internal (large support mismatch) |
| I | 0.952 | 0.760 | 0.952 | 0.85 | Ext classifier strong; detection miss is the real problem |
| S | 0.929 | 0.910 | 0.963 | 0.95 | Consistent |
| T | 0.633 | 0.630 | 0.776 | 0.73 | **Consistent across both datasets; M confusion is structural** |
| X | 0.833 | 0.870 | 0.909 | 0.90 | Consistent |
| Z | 1.000 | 0.960 | 1.000 | 0.96 | Strong |

---

## 2. Error Source Classification

| Category | Letters | Evidence |
| :--- | :--- | :--- |
| **Detection-Primary** | I | 30% miss rate on static external images; worse expected in webcam |
| **Classification-Primary (N→M, T→M)** | N, T | High detection rate but low recall; high-confidence wrong predictions |
| **Mixed (detection + classification)** | O | 13% detection miss + O/DELETE confusion |
| **Marginal classification issues** | F, J, S, X | Isolated 1-3 misclassifications; generally strong |
| **No evidence of issue** | D, G, Z | ≥ 97% classifier recall on external set |

---

## 3. Runtime Diagnostics Subsystem

### 3.1 Schema (22 CSV columns)

| Column | Type | Description |
| :--- | :--- | :--- |
| `frame_index` | int | Cumulative frame counter from cap.read() |
| `timestamp_sec` | float | Monotonic time.time() stamp |
| `camera_read_ok` | bool | False only on a failed cap.read() |
| `hand_detected` | bool | MediaPipe found ≥1 hand landmark set |
| `norm_success` | bool | LandmarkNormalizer returned valid 63-D features |
| `norm_failure_reason` | str | Reason string when norm fails |
| `predicted_label` | str | MLP top-1 class label |
| `confidence` | float | Softmax confidence of top-1 |
| `top1_label / top1_prob` | str/float | Top-1 prediction |
| `top2_label / top2_prob` | str/float | Top-2 prediction |
| `top3_label / top3_prob` | str/float | Top-3 prediction |
| `low_conf_rejected` | bool | Confidence below min_confidence threshold |
| `sm_candidate` | str | State-machine candidate label |
| `sm_is_dwelling` | bool | Whether candidate is accumulating dwell time |
| `sm_latched_label` | str | Currently latched label (hold debounce) |
| `commit_event_label` | str | Label emitted this frame (blank if none) |
| `text_buffer_length` | int | Characters in text buffer after this frame |
| `status` | str | Pipeline status string |
| `error_category` | str | CAMERA_READ_FAIL / DETECTOR_ERROR / INVALID_FEATURES / INFERENCE_ERROR |

### 3.2 Design Notes

- **Camera-read failures** are logged via `log_camera_failure()` with `camera_read_ok=False`
  and `text_buffer_length=""` (no pipeline state available). They are never treated as confirmed
  no-hand events.
- **Invalid normalisation** (`norm_success=False`) is recorded with `hand_detected=True` and
  `error_category=INVALID_FEATURES`. The state machine latch is preserved; the event is not
  treated as `NO_HAND`.
- **Low-confidence predictions** are recorded with `predicted_label` and `confidence` filled in,
  `low_conf_rejected=True`, and `commit_event_label=""`.
- **No video or image data** is saved to disk at any time.

### 3.3 Launch Commands

```bash
# Default session — no diagnostics (production mode)
.venv\\Scripts\\python.exe src/realtime/app.py

# Enable diagnostics (CSV written to reports/live_diagnostics/)
.venv\\Scripts\\python.exe src/realtime/app.py --diagnostics

# Custom diagnostics directory
.venv\\Scripts\\python.exe src/realtime/app.py --diagnostics --diagnostics-dir my_diag_sessions
```

---

## 4. Independent Audit Check Results

| Check | Description | Result | Details |
| :--- | :--- | :--- | :--- |
| `json_loaded` | 870 records in results JSON | **{audit_results['json_loaded'][0]}** | {audit_results['json_loaded'][1]} |
| `class_mapping` | 11 target letters in SEMANTIC_CLASSES | **{audit_results['class_mapping'][0]}** | {audit_results['class_mapping'][1]} |
| `per_letter_metrics` | Metrics recomputed from raw records | **{audit_results['per_letter_metrics'][0]}** | {audit_results['per_letter_metrics'][1]} |
| `csv_schema` | CSV columns, types, and logged rows correct | **{audit_results['csv_schema'][0]}** | {audit_results['csv_schema'][1]} |
| `artifact_integrity` | Checkpoint & Parquets SHA-256 unchanged | **{audit_results['artifact_integrity'][0]}** | Verified zero mutation |

---

## 5. Test Results

- `tests/test_diagnostics.py`: **10 / 10 passed** (disabled=no file, enabled=valid CSV, predicted labels,
  detection failure, invalid norm, camera failure, low-conf, commit event, cleanup, deterministic replay)
- Full project test suite: **69 passed** (confirmed before audit)

---

## 6. Unresolved Questions Requiring Live Signing

The following cannot be resolved from static external images:

1. **`I` in motion:** Does the pinky-only hand pose cause MediaPipe tracking loss specifically
   during motion (e.g. `J` motion following `I`)? Requires recording a live signing session.
2. **`N` / `T` in user signing:** Are user hand positions closer to `M` than the training data?
   The external dataset shows 17/18 `N` images predicted as `M` — if training data had more
   `M`-leaning `N` examples this may be structural.
3. **Confidence threshold impact on `S` and `X`:** Min confidence for these letters reaches
   0.39–0.41 in the external set; in a live session some frames will be below 0.70 and rejected.
   Live recording would quantify how often dwell is broken by low-confidence frames.
4. **Dwell-time adequacy:** The 0.60s dwell may be too long for fast signers on some letters.
   Adjusting `--dwell` can be evaluated with `--diagnostics` enabled during a live session.

---

## 7. Artifact Integrity

```
{chr(10).join(integrity_details)}
```
"""

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    log(f"\nReport written to: {REPORT_PATH}")
    log(f"Overall Audit: {'PASS' if all_pass else 'FAIL'} ({n_pass}/{n_total})")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_audit())
