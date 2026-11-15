# Real-Time American Sign Language (ASL) Recognition & Speech Synthesis System

[![Python 3.12](https://img.shields.io/badge/Python-3.12.10-blue.svg)](https://www.python.org/downloads/release/python-31210/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0%20(CPU%2FCUDA)-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0.1-brightgreen.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Tests-124%20passed%20(100%25)-success.svg)](tests/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end, production-oriented Computer Vision system that converts American Sign Language (ASL) fingerspelling and dynamic gestures into synthesized speech in real-time.

The application captures live webcam video, extracts 3D hand landmarks via MediaPipe, normalizes pose geometry into scale- and translation-invariant feature vectors, classifies static signs using a lightweight PyTorch MLP (~18.5k parameters), detects dynamic strokes ('J' and 'Z') via kinematic temporal analysis, stabilizes emissions using a Finite State Machine (FSM), and speaks completed words asynchronously through an offline Text-to-Speech (TTS) engine.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Directory Structure](#project-directory-structure)
- [Technology Stack](#technology-stack)
- [Installation & Environment Setup (Windows)](#installation--environment-setup-windows)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone Repository](#step-1-clone-repository)
  - [Step 2: Virtual Environment Setup](#step-2-virtual-environment-setup)
  - [Step 3: Dependency Installation](#step-3-dependency-installation)
- [Running the Application](#running-the-application)
  - [Launch Commands](#launch-commands)
  - [CLI Flags & Options](#cli-flags--options)
  - [Keyboard Controls & HUD Guide](#keyboard-controls--hud-guide)
- [Subsystems & Capabilities](#subsystems--capabilities)
  - [1. Static ASL Recognition (PyTorch MLP)](#1-static-asl-recognition-pytorch-mlp)
  - [2. Dynamic Gesture Recognition ('J' and 'Z')](#2-dynamic-gesture-recognition-j-and-z)
  - [3. Asynchronous Offline Text-to-Speech (TTS)](#3-asynchronous-offline-text-to-speech-tts)
  - [4. Robustness & Environmental Stress Evaluation](#4-robustness--environmental-stress-evaluation)
  - [5. Two-Hand Feature Representation (Feasibility Prototype)](#5-two-hand-feature-representation-feasibility-prototype)
- [Performance & Latency Profiling](#performance--latency-profiling)
- [Testing & Validation](#testing--validation)
- [Dataset Acquisition & Full Extraction](#dataset-acquisition--full-extraction)
- [Sign Language MNIST CNN Benchmark](#sign-language-mnist-cnn-benchmark)
- [Screenshots & UI Walkthrough](#screenshots--ui-walkthrough)
- [Limitations & Future Scope](#limitations--future-scope)
- [License](#license)

---

## Key Features

- **Real-Time 3D Landmark Tracking**: Tracks 21 hand landmarks (63 spatial coordinates) per frame at 40–55 FPS using MediaPipe Hand Landmarker.
- **Invariant Geometry Normalization**: Translation normalization (wrist origin), scale normalization (wrist-to-middle-MCP distance), and handedness canonicalization (left hands reflected onto right-hand coordinate frame).
- **Lightweight PyTorch MLP Classifier**: 3-layer architecture (63 &rarr; 128 &rarr; 64 &rarr; 28) with only 18,524 parameters, requiring **0.155 ms** CPU inference per frame.
- **Dynamic Gesture Recognition ('J' & 'Z')**: Kinematic trajectory buffer tracking fingertip motion vectors, curvature, and direction with zero static MLP interference.
- **Temporal Stabilization (FSM)**: Hysteresis state machine requiring sustained dwell (default 0.60s) and release debounce (0.25s) to eliminate jitter and duplicate letters.
- **Offline Text-to-Speech (TTS)**: Non-blocking worker thread powered by `pyttsx3` with Windows COM stabilization, speaking completed words on `SPACE` or following a 2.5s conversational pause.
- **Environmental Robustness Benchmark**: Reproducible synthetic stress framework evaluating lighting, Gaussian noise, motion blur, occlusions, scale, and tilt.
- **Multi-Hand Extension Framework**: 133-dimensional two-hand feature representation with explicit validity masks and inter-wrist relative geometry for future bimanual signs.

---

## System Architecture

```
[Webcam Stream]
      │
      ▼
[OpenCV Frame Capture] (Horizontal Mirroring)
      │
      ▼
[MediaPipe Hand Landmarker] (Extracts 21 3D landmarks @ ~20ms)
      │
      ├───► [Temporal Sequence Buffer] (Tracks 35-frame pinky/index trajectories)
      │         │
      │         ▼
      │     [Kinematic Dynamic Classifier] ──► Qualified 'J' or 'Z' Stroke Detected?
      │         │                                      │
      ▼         ▼                                      │ (Bypasses static dwell)
[LandmarkNormalizer]                                   │
  ├── Wrist Translation (origin at (0,0,0))            │
  ├── Scale Normalization (wrist-to-middle-MCP)        │
  └── Handedness Canonicalization (Left -> Right)      │
      │                                                │
      ▼                                                │
[PyTorch ASLLandmarkMLP] (63-D -> 28 classes)          │
      │                                                │
      ▼                                                │
[Confidence & Ambiguity Gate]                          │
      │                                                │
      ▼                                                │
[GestureStateMachine (FSM)] ◄──────────────────────────┘
  ├── Dwell Confirmation (0.60s sustained hold)
  └── Release Hysteresis (0.25s debounce before re-arming)
      │
      ▼
[TextBuffer] (Appends chars, handles SPACE / DELETE, avoids double spaces)
      │
      ├───► [SpeechEngine (TTS)] (Background worker, speaks on SPACE or 2.5s pause)
      │
      ▼
[OpenCV Visual HUD Display] (Renders camera feed, skeleton, probabilities, and text line)
```

---

## Project Directory Structure

```text
ASL-Gesture-Recognition/
├── docs/
│   └── FINAL_TECHNICAL_REPORT.md   # Complete architectural and empirical research report
├── models/
│   ├── asl_landmark_mlp_best.pt    # Trained PyTorch 28-class MLP weights (18,524 params)
│   └── mnist_cnn_best.pt           # Benchmark Sign Language MNIST CNN weights
├── reports/
│   ├── robustness/                 # Environmental stress and profiling JSON/MD reports
│   └── live_diagnostics/           # Optional CSV telemetry sessions
├── scripts/
│   ├── evaluate_robustness.py      # Reproducible robustness and latency profiling tool
│   ├── train_mlp.py                # MLP training script for ASL Alphabet
│   └── audit_live_diagnostics.py   # Diagnostics verification and telemetry audit
├── src/
│   ├── benchmark/                  # Sign Language MNIST CNN baseline module
│   ├── landmarks/
│   │   ├── labels.py               # 28 semantic class mappings (A-Z, SPACE, DELETE)
│   │   ├── normalizer.py           # Single-hand 63-D invariant normalization
│   │   └── two_hand.py             # Two-hand 133-D feature representation & prototype
│   ├── model/
│   │   └── mlp.py                  # PyTorch ASLLandmarkMLP neural network definition
│   └── realtime/
│       ├── app.py                  # Main real-time OpenCV desktop application
│       ├── gesture_state_machine.py# Temporal FSM for dwell confirmation & debounce
│       ├── inference.py            # PyTorch inference wrapper with class probabilities
│       ├── speech.py               # Asynchronous offline TTS engine (pyttsx3)
│       ├── temporal_sequence.py    # Kinematic trajectory buffer & J/Z dynamic classifier
│       └── text_buffer.py          # Word construction and editing buffer
├── tests/
│   ├── test_dynamic_gestures.py    # Unit tests for J/Z trajectories and kinematics
│   ├── test_mlp.py                 # PyTorch model architecture tests
│   ├── test_normalizer.py          # Geometric invariance unit tests
│   ├── test_realtime_inference.py  # Inference engine and probability tests
│   ├── test_realtime_pipeline.py   # Integrated pipeline execution tests
│   ├── test_realtime_state_machine.py # FSM dwell and hysteresis timing tests
│   ├── test_robustness.py          # Robustness transformations and metric tests
│   ├── test_speech.py              # Asynchronous TTS queue and worker tests
│   ├── test_text_buffer.py         # Text assembly and backspacing tests
│   └── test_two_hand.py            # Two-hand extraction, mask, and prototype tests
├── hand_landmarker.task            # MediaPipe Hand Landmarker model bundle (7.46 MB)
├── requirements.txt                # Dependencies for CUDA 13.2 GPU environments
├── requirements-cpu.txt            # Dependencies for CPU-only environments
├── pytest.ini                      # Pytest discovery configuration
└── README.md                       # Main project documentation
```

---

## Technology Stack

- **Language:** Python 3.12.10
- **Computer Vision:** OpenCV (`opencv-python==5.0.0.93`), MediaPipe (`mediapipe==1.0.1`)
- **Deep Learning:** PyTorch (`torch==2.14.0+cu132` / CPU fallback)
- **Audio & Speech:** `pyttsx3>=2.90` (Offline SAPI5 on Windows), `comtypes`
- **Scientific Computing:** NumPy (`2.5.2`), SciPy (`1.18.1`), scikit-learn (`1.9.1`)
- **Testing & Profiling:** Pytest (`9.1.1`), Python `time.perf_counter`

---

## Installation & Environment Setup (Windows)

### Prerequisites

- Windows 10/11 (64-bit)
- Python 3.12.10 installed and added to system `PATH`
- Working webcam and audio speakers/headphones

### Step 1: Clone Repository

```powershell
git clone https://github.com/Gaurav-Ojha65/ASL-Gesture-Recognition.git
cd ASL-Gesture-Recognition
```

### Step 2: Virtual Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Dependency Installation

For standard CPU execution (recommended):
```powershell
pip install -r requirements-cpu.txt
```

For NVIDIA GPU / CUDA 13.2 execution:
```powershell
pip install -r requirements.txt
```

---

## Running the Application

### Launch Commands

Start the real-time desktop application with all features enabled (static alphabet, dynamic J/Z, and speech):
```powershell
python src/realtime/app.py
```

### CLI Flags & Options

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--camera` | `0` | System webcam device ID / camera index (e.g. `0`, `1`). |
| `--dwell` | `0.60` | Sustained dwell duration in seconds required to commit a letter. |
| `--release` | `0.25` | Sustained absence required to re-arm the same letter. |
| `--min-conf` | `0.70` | Minimum softmax confidence threshold. |
| `--no-dynamic` | `False` | Disables dynamic J/Z gesture recognition. |
| `--no-speech` | `False` | Disables text-to-speech audio synthesis. |
| `--no-mirror` | `False` | Disables horizontal mirroring of webcam display. |
| `--diagnostics` | `False` | Enables frame-by-frame telemetry logging to CSV. |
| `--max-frames` | `None` | Optional limit on processed frames (useful for headless testing). |

### Keyboard Controls & HUD Guide

- **`V`**: Toggle Text-to-Speech **ON / OFF** in real time.
- **`C`**: Clear text buffer.
- **`Q` or `ESC`**: Gracefully shut down application and background worker threads.

---

## Subsystems & Capabilities

### 1. Static ASL Recognition (PyTorch MLP)
- Recognizes 26 uppercase alphabet letters (A–Z) plus control gestures `SPACE` and `DELETE`.
- Trained on 87,000+ hand landmark vectors with translation and scale invariance.
- Runs on CPU with **0.155 ms** inference latency.

### 2. Dynamic Gesture Recognition ('J' and 'Z')
- **Letter J:** Recognizes the physical curved stroke performed with the extended pinky finger.
- **Letter Z:** Recognizes the three-stroke zig-zag motion performed with the extended index finger.
- Incorporates a 1.2-second cooldown to prevent duplicate triggers.
- Automatically remaps stationary MLP predictions for J &rarr; I and Z &rarr; D, eliminating false static commits.

### 3. Asynchronous Offline Text-to-Speech (TTS)
- Completely non-blocking: Enqueue overhead on the camera thread is only **0.025 ms**.
- **Word Completion:** Automatically speaks the completed word when `SPACE` is committed.
- **Conversational Pause:** Speaks buffered words after a configurable 2.5-second pause.
- Duplicate suppression prevents repeated utterances within a 4.0-second window.

### 4. Robustness & Environmental Stress Evaluation
Run the reproducible evaluation script:
```powershell
python scripts/evaluate_robustness.py
```
Outputs performance benchmarks and environmental stress tests across brightness, noise, motion blur, and occlusions to `reports/robustness/`.

### 5. Two-Hand Feature Representation (Feasibility Prototype)
Implemented in [`src/landmarks/two_hand.py`](file:///d:/ASL_cv/ASL-Gesture-Recognition/src/landmarks/two_hand.py):
- Extracts up to two hands with deterministic Left/Right sorting.
- Produces a **133-dimensional** vector (63 left + 63 right + 5 inter-hand spatial + 2 validity masks).
- Feature normalization takes **0.028 ms** on CPU.
- *Status:* Architectural foundation and heuristic prototype; ready for future multi-hand trained models.

---

## Performance & Latency Profiling

Benchmarked on **Intel Core i7-12700H (12 Cores, Windows 11, PyTorch CPU mode)**:

| Pipeline Component | Mean Latency (ms) | Median (ms) | P95 (ms) | Real-Time Budget Share |
| :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Landmark Tracking** | 18.50 ms | 18.20 ms | 22.10 ms | ~98.0% |
| **Landmark Normalization (63-D)** | 0.012 ms | 0.012 ms | 0.012 ms | <0.1% |
| **PyTorch MLP Inference (28 cls)** | 0.155 ms | 0.099 ms | 0.327 ms | ~0.8% |
| **Dynamic Trajectory Classifier (J/Z)**| 0.035 ms | 0.032 ms | 0.052 ms | ~0.2% |
| **Temporal FSM & Text Buffer** | 0.001 ms | 0.001 ms | 0.001 ms | <0.01% |
| **Total Post-Perception Processing** | **0.203 ms** | **0.144 ms** | **0.392 ms** | **<1.0%** |
| **TTS Queue Latency (Main Thread)** | **0.025 ms** | **0.014 ms** | **0.085 ms** | Non-blocking |

---

## Testing & Validation

Execute the automated test suite across all subsystems:
```powershell
python -m pytest tests -v
```

### Test Suite Summary
- **Total Tests:** 124
- **Passed:** **124 passed (100%)** in ~6.5 seconds.
- **Failures:** 0. All subsystems—normalization, PyTorch inference, temporal FSM, text buffering, offline speech synthesis, dynamic J/Z trajectory kinematics, robustness transforms, and two-hand feature extraction—pass automated regression verification.

---

## Dataset Acquisition & Full Extraction

Datasets are excluded from Git to keep the repository lightweight. If you wish to reproduce landmark extraction and train the MLP from scratch:

1. **ASL Alphabet Dataset (Kaggle):**
   - Download the raw dataset: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
   - Extract raw images to:
     ```
     data/raw/asl_alphabet/asl_alphabet_train/
     ```
2. **Execute Full Landmark Extraction:**
   ```powershell
   python full_extraction.py
   ```
   *Extracts 21 3D landmarks for 87,000 images, normalizes coordinates, and writes train/val/test splits to Parquet files under `data/processed/landmarks/full/`.*

3. **Train the PyTorch MLP:**
   ```powershell
   python scripts/train_mlp.py
   ```

---

## Sign Language MNIST CNN Benchmark

An isolated 2D CNN module is included for evaluating pixel-based classification on the 28x28 grayscale Sign Language MNIST benchmark:

1. **Dataset Setup:**
   - Download CSVs from: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
   - Place files in: `data/raw/sign_language_mnist/`
2. **Train the Benchmark CNN:**
   ```powershell
   python src/benchmark/train.py
   ```
3. **Evaluate the Benchmark CNN:**
   ```powershell
   python src/benchmark/evaluate.py
   ```

---

## Screenshots & UI Walkthrough

*(Run the application locally to observe live feed with HUD overlay)*

```
+-----------------------------------------------------------+
| Real-Time ASL Recognition HUD                 FPS: 45.2   |
| Speech: [ON] (Press 'V' to toggle)                        |
|                                                           |
|                 [Hand Skeleton Overlay]                   |
|                      o---o---o                            |
|                     /                                     |
|                                                           |
| Top Prediction: [L]  (94.2%)                              |
| Dwell Progress: [===============>      ] 72%              |
|                                                           |
| Committed Text: HELLO WORLD_                              |
+-----------------------------------------------------------+
```

---

## Limitations & Future Scope

### Current Limitations
1. **Static MLP Trained on Single Hands:** The current MLP operates strictly on 63-dimensional single-hand landmarks. Two-handed signs require a dedicated multi-hand model.
2. **Heuristic J/Z Dynamic Recognition:** Stroke classification for 'J' and 'Z' uses explainable geometric kinematic rules rather than a deep sequence model (e.g. LSTM/GRU).
3. **Lighting & Contrast Sensitivity:** Hand landmark accuracy degrades under severe underexposure (<15 lux) or excessive backlight.

### Future Scope
1. **Sequence Deep Learning:** Train an LSTM or Temporal Convolutional Network (TCN) on continuous signing videos.
2. **WLASL Two-Hand Integration:** Train a dedicated two-hand model on the 133-D feature representation using the WLASL dataset.
3. **Word Autocompletion & LLM Correction:** Integrate n-gram language models to correct typos and expand signed abbreviations into natural sentences.

---

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
