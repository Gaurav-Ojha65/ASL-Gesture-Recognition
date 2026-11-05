"""
STEP 5.1 — ASL Alphabet Dataset Inspection & MediaPipe Pilot
Does NOT train MLP, normalize, or process all images.
"""

import os
import sys
import glob
import random
import json
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Paths ──
DATA_DIR = os.path.join("data", "raw", "asl_alphabet", "asl_alphabet_train", "asl_alphabet_train")
if not os.path.exists(DATA_DIR):
    # Fallback to single level if nested isn't correct
    DATA_DIR = os.path.join("data", "raw", "asl_alphabet", "asl_alphabet_train")

REPORTS_DIR = "reports"
PILOT_LANDMARKS_DIR = os.path.join("data", "processed", "landmarks", "pilot")
PILOT_EXAMPLES_DIR = os.path.join(REPORTS_DIR, "pilot_examples")
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PILOT_LANDMARKS_DIR, exist_ok=True)
os.makedirs(PILOT_EXAMPLES_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

def log(msg=""):
    print(msg)
    report_lines.append(msg)

report_lines = []

# ═══════════════════════════════════════════════════════════════
# TASK 1 & 2 — DATASET INSPECTION & CLASSES
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 1 & 2 — DATASET INSPECTION")
log("=" * 70)

# 29 Semantic classes
semantic_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['DELETE', 'NOTHING', 'SPACE']
# Get physical folders
folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

log(f"Number of physical folders: {len(folders)}")

# Mapping
physical_to_semantic = {}
for folder in folders:
    upper = folder.upper()
    if upper == "DEL":
        semantic = "DELETE"
    elif upper in semantic_classes:
        semantic = upper
    else:
        semantic = upper
    physical_to_semantic[folder] = semantic

log("Class Mapping & Image Counts:")
class_stats = {}
total_images = 0
for folder in folders:
    semantic = physical_to_semantic[folder]
    folder_path = os.path.join(DATA_DIR, folder)
    # Using multiple extensions just in case, typically they are .jpg
    images = glob.glob(os.path.join(folder_path, "*.*"))
    # Filter to actual images
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    class_stats[semantic] = {
        'physical': folder,
        'count': len(images),
        'images': sorted(images) # deterministic sort
    }
    total_images += len(images)
    log(f"  {folder:10s} -> {semantic:10s} : {len(images)} images")

log(f"\nTotal semantic classes expected: 29")
log(f"Total semantic classes found: {len(class_stats)}")
log(f"Total images found: {total_images}")

# Check formats and dimensions for a sample
sample_img_path = class_stats['A']['images'][0]
sample_img = cv2.imread(sample_img_path)
if sample_img is not None:
    h, w, c = sample_img.shape
    log(f"Sample image dimensions: {w}x{h}x{c}")
else:
    log("Warning: Could not read sample image")

log()

# ═══════════════════════════════════════════════════════════════
# TASK 3 — PILOT SELECTION
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 3 — PILOT SELECTION (2900 IMAGES)")
log("=" * 70)

pilot_images = []
for semantic, stats in class_stats.items():
    imgs = stats['images']
    # shuffle deterministically
    random.seed(SEED) # Reseed per class is fine or just rely on global, let's just use global
    imgs_copy = list(imgs)
    random.shuffle(imgs_copy)
    selected = imgs_copy[:100] if len(imgs_copy) >= 100 else imgs_copy
    pilot_images.extend([(semantic, img) for img in selected])

log(f"Total pilot images selected: {len(pilot_images)}")
log()

# ═══════════════════════════════════════════════════════════════
# TASK 4 & 5 & 6 — MEDIAPIPE EXTRACTION & REPORTS & RAW LANDMARKS
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 4 & 5 & 6 — MEDIAPIPE EXTRACTION")
log("=" * 70)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print(f"Downloading MediaPipe model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

detection_stats = {semantic: {'total': 0, 'success': 0, 'failed': 0} for semantic in class_stats.keys()}

print("Running MediaPipe extraction...")

# for visualization examples
examples_saved = 0
classes_for_examples = set(['A', 'DELETE', 'NOTHING', 'SPACE', 'M', 'N', 'Z', 'J']) # Pick a few diverse ones

for idx, (semantic, img_path) in enumerate(pilot_images):
    detection_stats[semantic]['total'] += 1

    # Load image via MediaPipe
    mp_image = mp.Image.create_from_file(img_path)

    # Detect
    detection_result = detector.detect(mp_image)

    if len(detection_result.hand_landmarks) > 0:
        detection_stats[semantic]['success'] += 1

        hand_landmarks = detection_result.hand_landmarks[0]
        handedness = detection_result.handedness[0][0]

        # Save raw landmarks
        landmark_data = {
            "source_image": img_path,
            "class": semantic,
            "handedness": {
                "category": handedness.category_name,
                "score": handedness.score
            },
            "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
        }

        base_name = os.path.basename(img_path)
        out_json_path = os.path.join(PILOT_LANDMARKS_DIR, f"{base_name}.json")
        with open(out_json_path, 'w') as f:
            json.dump(landmark_data, f, indent=2)

        # Task 7: Diagnostic Visualization
        if semantic in classes_for_examples and examples_saved < 15:
            # Draw landmarks on image
            annotated_image = mp_image.numpy_view().copy()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            h, w, _ = annotated_image.shape

            # Draw landmarks manually
            connections = [
                (0,1), (1,2), (2,3), (3,4),       # Thumb
                (0,5), (5,6), (6,7), (7,8),       # Index
                (5,9), (9,10), (10,11), (11,12),  # Middle
                (9,13), (13,14), (14,15), (15,16),# Ring
                (13,17), (17,18), (18,19), (19,20),# Pinky
                (0,17)                            # Palm base
            ]

            pts = []
            for lm in hand_landmarks:
                pts.append((int(lm.x * w), int(lm.y * h)))

            for pt in pts:
                cv2.circle(annotated_image, pt, 2, (0, 0, 255), -1)

            for (p1, p2) in connections:
                if 0 <= p1 < len(pts) and 0 <= p2 < len(pts):
                    cv2.line(annotated_image, pts[p1], pts[p2], (0, 255, 0), 2)

            cv2.imwrite(os.path.join(PILOT_EXAMPLES_DIR, f"detected_{semantic}_{base_name}"), annotated_image)
            classes_for_examples.remove(semantic) # Just one per requested class
            examples_saved += 1

    else:
        detection_stats[semantic]['failed'] += 1

        # Save failed example for visualization if from a requested class
        if semantic in classes_for_examples and examples_saved < 15:
            annotated_image = cv2.imread(img_path)
            cv2.imwrite(os.path.join(PILOT_EXAMPLES_DIR, f"failed_{semantic}_{os.path.basename(img_path)}"), annotated_image)
            classes_for_examples.remove(semantic)
            examples_saved += 1

    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1} / {len(pilot_images)}...")

total_processed = sum(s['total'] for s in detection_stats.values())
total_success = sum(s['success'] for s in detection_stats.values())
total_failed = sum(s['failed'] for s in detection_stats.values())
overall_rate = total_success / total_processed if total_processed > 0 else 0

log(f"Total images processed: {total_processed}")
log(f"Successful detections:  {total_success}")
log(f"Failed detections:      {total_failed}")
log(f"Overall detection rate: {overall_rate * 100:.2f}%")
log()

log("Per-class Detection Report:")
log(f"{'Class':<10} | {'Total':<6} | {'Success':<8} | {'Failed':<7} | {'Rate':<7}")
log("-" * 46)

classes_list = []
rates_list = []

for semantic in sorted(detection_stats.keys()):
    stats = detection_stats[semantic]
    t = stats['total']
    s = stats['success']
    f = stats['failed']
    r = s / t if t > 0 else 0
    log(f"{semantic:<10} | {t:<6} | {s:<8} | {f:<7} | {r*100:.2f}%")
    classes_list.append(semantic)
    rates_list.append(r * 100)

log()

# ═══════════════════════════════════════════════════════════════
# TASK 5 (PLOT)
# ═══════════════════════════════════════════════════════════════
plt.figure(figsize=(14, 6))
bars = plt.bar(classes_list, rates_list, color='skyblue')
plt.axhline(y=overall_rate*100, color='r', linestyle='-', label=f'Overall Avg ({overall_rate*100:.1f}%)')
plt.title('MediaPipe Hand Detection Rate by Class (Pilot: 100 imgs/class)')
plt.ylabel('Detection Rate (%)')
plt.xlabel('Semantic Class')
plt.xticks(rotation=45)
plt.ylim(0, 110)
plt.legend()

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.0f}%', va='bottom', ha='center', fontsize=8, rotation=90)

plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "asl_alphabet_pilot_detection.png")
plt.savefig(plot_path)
plt.close()

# ═══════════════════════════════════════════════════════════════
# WRITE REPORT
# ═══════════════════════════════════════════════════════════════
md = "# ASL Alphabet Pilot & Dataset Inspection Report\n\n"
for line in report_lines:
    if line.startswith("=" * 70):
        md += "---\n"
    elif line.startswith("TASK"):
        md += f"\n## {line}\n\n"
    elif line == "":
        md += "\n"
    else:
        md += f"{line}  \n"

report_path = os.path.join(REPORTS_DIR, "asl_alphabet_pilot_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Report saved to {report_path}")
print(f"Plot saved to {plot_path}")
print(f"Raw landmarks saved to {PILOT_LANDMARKS_DIR}")
print(f"Examples saved to {PILOT_EXAMPLES_DIR}")
