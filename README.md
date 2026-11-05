# Real-Time American Sign Language (ASL) Gesture Recognition Interface

A real-time American Sign Language (ASL) fingerspelling recognition system that processes live webcam video, extracts 3D hand landmarks via MediaPipe, canonicalizes and normalizes hand geometry, classifies signs with a lightweight PyTorch Multi-Layer Perceptron (MLP), and stabilizes character emissions into words and sentences using a temporal Finite State Machine (FSM).

An isolated benchmark module is also included for evaluating a custom 2D Convolutional Neural Network (CNN) on the Sign Language MNIST dataset.

---

## Table of Contents

- [Features](#features)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Project Directory Structure](#project-directory-structure)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Dependency Installation](#dependency-installation)
  - [Option A: NVIDIA GPU / CUDA 13.2 (Recommended)](#option-a-nvidia-gpu--cuda-132-recommended)
  - [Option B: CPU-Only / Cross-Platform](#option-b-cpu-only--cross-platform)
- [Required Model Assets & Checkpoints](#required-model-assets--checkpoints)
- [Running the Real-Time Application](#running-the-real-time-application)
  - [CLI Flags & Configuration](#cli-flags--configuration)
  - [Keyboard Controls](#keyboard-controls)
- [Model Training & Evaluation](#model-training--evaluation)
  - [1. ASL Landmark MLP (Live Pipeline)](#1-asl-landmark-mlp-live-pipeline)
  - [2. Sign Language MNIST Benchmark (CNN)](#2-sign-language-mnist-benchmark-cnn)
- [Running the Test Suite](#running-the-test-suite)
- [Dataset Acquisition & Directory Setup](#dataset-acquisition--directory-setup)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Features

- **Live Hand Landmark Tracking**: Tracks 21 3D hand landmarks in real time using the MediaPipe Hand Landmarker Task.
- **Invariant Preprocessing**: Translates coordinates relative to the wrist (landmark 0), scales for camera distance invariance, and canonicalizes left/right handedness to ensure signs are recognized regardless of which hand is used.
- **Lightweight PyTorch MLP**: Efficient 3-layer architecture (63 inputs &rarr; 128 hidden &rarr; 64 hidden &rarr; 28 outputs) requiring only ~18.5k parameters and running inference in under 2 ms on CPU.
- **29 Semantic Classes**: Recognizes letters **A–Z**, plus control gestures: **SPACE**, **DELETE** (backspace), and **NOTHING** (idle / non-signing state).
- **Temporal Finite State Machine (FSM)**: Eliminates gesture flickering and unintended repeated letters with configurable dwell duration (default: 0.60s) and release debounce hysteresis (default: 0.25s).
- **Text Accumulation Buffer**: Assembles recognized characters into words, preventing double spaces and handling backspacing seamlessly.
- **Visual HUD Overlay**: Real-time Heads-Up Display featuring bounding boxes, skeleton wireframe, top-3 class probabilities, FSM dwell progress indicator, and accumulated text line.
- **Production Safety & Diagnostics**: Per-session telemetry recording (`--diagnostics`) saves frame-by-frame state to CSV for offline analysis without capturing webcam images.

---

## System Architecture & Pipeline

The live application operates as a sequential, deterministic pipeline:

```
[Webcam Stream]
      │
      ▼
[OpenCV Frame Capture] (Optional horizontal mirroring)
      │
      ▼
[MediaPipe Hand Landmarker] (Extracts 21 3D raw landmarks)
      │
      ▼
[LandmarkNormalizer]
  ├── Handedness Canonicalization (flips x for left hand to right-hand canonical frame)
  ├── Wrist Centering (translates all 21 points relative to landmark 0)
  ├── Scale Normalization (divides coordinates by max euclidean distance from wrist)
  └── Vector Flattening (produces clean 63-dimensional feature vector)
      │
      ▼
[PyTorch ASLLandmarkMLP] (Softmax over 28 trained hand gesture classes)
      │
      ▼
[Confidence Gate] (Discards predictions below --min-conf, default: 0.70)
      │
      ▼
[GestureStateMachine]
  ├── Candidate Tracking (tracks consistent predictions over time)
  ├── Dwell Commitment (triggers CommitEvent only after sign is held for --dwell seconds)
  └── Release Hysteresis (requires hand release/transition for --release seconds before re-arming)
      │
      ▼
[TextBuffer] (Appends character, handles SPACE or DELETE, prevents consecutive spaces)
      │
      ▼
[OpenCV Visual HUD Display] (Renders camera feed, skeleton, diagnostics, and text)
```

> [!NOTE]
> **Architectural Separation:** The Sign Language MNIST CNN module under `src/benchmark/` is an isolated evaluation baseline for pixel-based 28x28 grayscale images. It is completely independent of the real-time MediaPipe landmark pipeline.

---

## Project Directory Structure

```text
Computer_Vision/
├── .gitignore                      # Git exclusion rules (ignores raw datasets, venv, caches)
├── pytest.ini                      # Pytest discovery configuration (sets pythonpath = .)
├── README.md                       # Comprehensive project documentation
├── SPEC.md                         # Architectural specifications and module boundaries
├── requirements.txt                # Full project dependencies (CUDA 13.2 / GPU default)
├── requirements-cpu.txt            # Project dependencies for CPU-only systems
├── hand_landmarker.task            # MediaPipe Hand Landmarker task model asset (7.46 MB)
│
├── src/                            # Core application source code
│   ├── benchmark/                  # Sign Language MNIST CNN baseline module
│   │   ├── cnn.py                  # PyTorch Conv2D architecture
│   │   ├── dataset.py              # PyTorch Dataset for MNIST CSVs
│   │   ├── data_loader.py          # DataLoader constructor
│   │   ├── train.py                # CNN training script
│   │   └── evaluate.py             # CNN evaluation and metrics export
│   ├── landmarks/                  # Preprocessing & landmark normalization
│   │   ├── labels.py               # 29 semantic class definitions & mapping indices
│   │   └── normalizer.py           # Wrist-centering, scaling & handedness canonicalizer
│   ├── model/                      # ASL Landmark MLP model components
│   │   ├── mlp.py                  # ASLLandmarkMLP neural network architecture
│   │   ├── landmark_dataset.py     # Parquet dataset reader for landmark vectors
│   │   ├── data_loader.py          # DataLoader constructor with shuffling & batching
│   │   └── trainer.py              # MLPTrainer with early stopping and loss logging
│   ├── realtime/                   # Real-time webcam pipeline
│   │   ├── app.py                  # Primary application entry point & OpenCV HUD
│   │   ├── inference.py            # Frozen ASLInferenceEngine wrapper
│   │   ├── gesture_state_machine.py# Temporal FSM for dwell & release debouncing
│   │   ├── text_buffer.py          # Character & word assembly text buffer
│   │   └── __init__.py
│   └── utils/                      # Helper utilities
│
├── models/                         # Pretrained model checkpoints
│   ├── asl_landmark_mlp_best.pt    # 28-class MLP weights (230 KB, required for live app)
│   └── mnist_cnn_best.pt           # Sign Language MNIST CNN weights (4.54 MB)
│
├── scripts/                        # Production utilities, training & diagnostic audits
│   ├── train_mlp.py                # Trains the MLP on extracted landmark Parquet files
│   ├── evaluate_external_dataset.py# Evaluates MLP against external ASL validation set
│   ├── audit_live_diagnostics.py   # Analyzes session CSV logs from --diagnostics
│   └── ...                         # Verification and milestone audit tools
│
├── tests/                          # Automated pytest suite (80 unit & integration tests)
│   ├── test_normalizer.py          # Normalizer invariance and boundary condition tests
│   ├── test_realtime_inference.py  # Inference engine architecture & prediction tests
│   ├── test_realtime_state_machine.py # State machine dwell, release & hysteresis tests
│   ├── test_realtime_pipeline.py   # End-to-end pipeline integration & replay tests
│   ├── test_text_buffer.py         # TextBuffer manipulation & boundary tests
│   └── test_diagnostics.py         # Telemetry logger tests
│
├── reports/                        # Curated evaluation metrics, markdown audits, and plots
│   └── live_diagnostics/           # Target folder for runtime session CSV logs
│
└── data/                           # Datasets directory (excluded from Git)
    ├── raw/                        # Raw image datasets (ASL Alphabet, Sign MNIST)
    └── processed/                  # Extracted landmark Parquet files
```

---

## Prerequisites

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python**: Version **3.12.x** (tested with Python 3.12.10)
- **Hardware**:
  - Webcam / USB Camera (for real-time recognition)
  - NVIDIA GPU with CUDA 12/13 support (optional; CPU is fully supported for inference)

---

## Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Computer_Vision
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

---

## Dependency Installation

### Option A: NVIDIA GPU / CUDA 13.2 (Recommended for CUDA Users)
If you have an NVIDIA GPU configured with CUDA 13.2:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: CPU-Only / Cross-Platform
If running on CPU, macOS (Apple Silicon), or Linux without CUDA:
```bash
pip install --upgrade pip
pip install -r requirements-cpu.txt
```

> [!TIP]
> Both options install the required `pyarrow` library for reading Parquet landmark files, OpenCV, MediaPipe, and PyTorch.

---

## Required Model Assets & Checkpoints

All model assets required to run the live application and tests are included directly in the repository:

1. **`hand_landmarker.task`** (7.46 MB, in project root):
   Pretrained MediaPipe vision task asset for 21 3D hand landmark extraction.
2. **`models/asl_landmark_mlp_best.pt`** (230.4 KB):
   Trained weights for the 28-class `ASLLandmarkMLP` classifier. Required for `app.py` and unit tests.
3. **`models/mnist_cnn_best.pt`** (4.54 MB):
   Trained weights for the Sign Language MNIST CNN benchmark evaluator.

---

## Running the Real-Time Application

Launch the real-time webcam interface with default settings:

```bash
python src/realtime/app.py
```

### CLI Flags & Configuration

You can customize camera selection, model checkpoints, and state machine timing via command-line arguments:

```bash
python src/realtime/app.py [OPTIONS]
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--camera` | `0` | Camera device index (e.g., `0`, `1`, `2`) |
| `--checkpoint` | `models/asl_landmark_mlp_best.pt` | Path to PyTorch MLP checkpoint file |
| `--task-model` | `hand_landmarker.task` | Path to MediaPipe `hand_landmarker.task` |
| `--dwell` | `0.60` | Dwell duration (in seconds) required to confirm a sign |
| `--release` | `0.25` | Release duration (in seconds) required before allowing the same sign again |
| `--min-conf` | `0.70` | Minimum prediction confidence threshold (0.0 to 1.0) |
| `--no-mirror` | `False` | Disable horizontal video feed mirroring |
| `--diagnostics` | `False` | Enable per-frame diagnostic CSV logging in `reports/live_diagnostics/` |
| `--max-frames` | `None` | Optional limit on processed frames (useful for headless testing) |

### Keyboard Controls

While the webcam window is active:
- **`Q`** or **`ESC`**: Safely exit the application and release camera resources.
- **`C`**: Clear the accumulated text buffer.

---

## Model Training & Evaluation

### 1. ASL Landmark MLP (Live Pipeline)

1. **Train the MLP model** (requires extracted Parquet files in `data/processed/landmarks/full/`):
   ```bash
   python scripts/train_mlp.py
   ```
   *Trains for up to 30 epochs with early stopping (patience = 7). Checkpoint is saved to `models/asl_landmark_mlp_best.pt`, and training curves are saved to `reports/asl_mlp_training_curves.png`.*

2. **Evaluate on external dataset**:
   ```bash
   python scripts/evaluate_external_dataset.py
   ```
   *Generates classification metrics and confusion matrix under `reports/`.*

### 2. Sign Language MNIST Benchmark (CNN)

1. **Train the benchmark CNN** (requires `sign_mnist_train.csv` and `sign_mnist_test.csv` in `data/raw/sign_language_mnist/`):
   ```bash
   python src/benchmark/train.py
   ```

2. **Evaluate the benchmark CNN**:
   ```bash
   python src/benchmark/evaluate.py
   ```

---

## Running the Test Suite

The project includes an automated test suite with **80 unit and integration tests** covering normalization invariance, inference determinism, state machine transitions, text buffering, and pipeline replay.

To run all tests using `pytest`:

```bash
pytest
```

Or execute via Python's module runner:

```bash
python -m pytest -v
```

> [!NOTE]
> Thanks to `pytest.ini`, both `pytest` and `python -m pytest` automatically resolve the project root on `sys.path`.

---

## Dataset Acquisition & Directory Setup

Datasets are excluded from Git to keep the repository lightweight. If you wish to reproduce full landmark extraction and model training from scratch:

### 1. ASL Alphabet Dataset (Images)
- Sourced from Kaggle: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Extract the training images into:
  ```
  data/raw/asl_alphabet/asl_alphabet_train/
  ```
- Run full landmark extraction into Parquet:
  ```bash
  python full_extraction.py
  ```

### 2. Sign Language MNIST Dataset (CSVs)
- Sourced from Kaggle: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Place the two CSV files into:
  ```
  data/raw/sign_language_mnist/sign_mnist_train.csv
  data/raw/sign_language_mnist/sign_mnist_test.csv
  ```

---

## Known Limitations

- **Single Hand Focus**: The current pipeline processes the primary detected hand. Multi-hand two-handed ASL signs are not currently supported.
- **Lighting & Background Sensitivity**: MediaPipe hand detection operates best in even lighting with high contrast against the background.
- **Static vs. Dynamic Signs**: Letters 'J' and 'Z' require dynamic movement in authentic ASL. The current static classifier recognizes standard canonical poses; temporal motion trajectory recognition is an area for future extension.
- **Benchmark vs. Live Application**: The Sign Language MNIST CNN module is an independent benchmark and cannot directly accept live camera frames due to format differences (28x28 grayscale vs 3D landmark features).

---

## License

This project is licensed under the Apache 2.0 License.
