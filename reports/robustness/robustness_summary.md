# ASL Gesture Recognition — Robustness and Performance Evaluation Report

**Generated:** 2026-11-15 22:08:17  
**Python:** 3.12.10 | **Platform:** Windows-11-10.0.26200-SP0  
**Random Seed:** 42

---

## 1. Latency & Performance Breakdown

| Pipeline Component | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Min / Max (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Landmark Normalization** | 0.012 | 0.012 | 0.012 | 0.012 / 0.012 |
| **Pytorch Mlp Inference** | 0.155 | 0.099 | 0.327 | 0.091 / 0.788 |
| **Dynamic Trajectory Evaluation** | 0.035 | 0.032 | 0.052 | 0.032 / 0.065 |
| **State Machine And Buffer** | 0.001 | 0.001 | 0.001 | 0.000 / 0.009 |

- **Total Post-Perception Latency (CPU):** **0.203 ms** per frame.
- **TTS Enqueue Overhead on Main Loop:** **0.0251 ms** (completely non-blocking).

## 2. Environmental Degradation Analysis

| Environmental Condition | Tested Transformation | Retention / Detection Rate | Notes |
| :--- | :--- | :---: | :--- |
| **baseline** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **brightness_dark_0.5x** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **brightness_bright_1.5x** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **gaussian_noise_sigma15** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **gaussian_noise_sigma30** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **motion_blur_k5** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **motion_blur_k9** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **occlusion_corner_15pct** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **occlusion_center_20pct** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **scale_far_0.75x** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **scale_close_1.25x** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **rotation_tilt_15deg** | Seed=42 | 100.0% | Evaluated on 6 samples |
| **rotation_tilt_neg15deg** | Seed=42 | 100.0% | Evaluated on 6 samples |

## 3. Scientific Distinctions & Methodology Note

> [!IMPORTANT]
> **Experimental Distinctions:**
> 1. **Baseline Benchmarks:** Trained MLP achieved 87.70% on held-out internal test set (4,399 samples) and 99.75% on Sign Language MNIST.
> 2. **Synthetic Degradations:** Synthetic noise, brightness scaling, and motion blur simulate environmental variations for stress testing.
> 3. **Physical Real-World Accuracy:** Real-world performance depends on physical webcam lighting, user signing fluency, and camera contrast.
