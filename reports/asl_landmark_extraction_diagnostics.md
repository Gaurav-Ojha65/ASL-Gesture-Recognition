# ASL Landmark Extraction Diagnostics

## Architectural Correction
**Important finding:** The `NOTHING` class in the dataset does not contain hands. MediaPipe correctly produces 0% detection for this class.
Instead of zero-filling landmarks (which creates artificial and meaningless data for a spatial MLP), the production MLP should only classify the 28 hand-bearing classes (`A-Z`, `SPACE`, `DELETE`).

**Proposed Architecture Update:**
- **State 1 (No Hand):** MediaPipe returns no detection. The system state is set to `NOTHING`.
- **State 2 (Hand Detected):** MediaPipe returns 21 landmarks. The coordinates are normalized and passed to the 28-class MLP for letter prediction.

*Note: This architecture has NOT been implemented yet per Step 5.2 constraints, but is strictly recommended.*

## Experiment Design
A deterministic subset of 10 difficult classes (M, N, SPACE, DELETE, P, X, C, Q, Z, J) with 100 images each was evaluated to identify whether preprocessing improves detection rates without introducing degenerate data.

- **Baseline:** 200x200 original resolution, 0.5 confidence threshold.
- **Upscale:** 400x400 bicubic resize, 0.5 confidence threshold.
- **Lower Threshold (T=0.3):** 200x200, 0.3 confidence threshold.
- **Upscale + Threshold (Up+T=0.3):** 400x400 bicubic resize, 0.3 confidence threshold.

*Quality Criteria:* Detections are only counted if 21 landmarks are returned, coordinates are finite, bounding constraints are met, and handedness is valid.

## Class-Wise Comparison

| Class    | Baseline (%) | Upscale (%) | T=0.3 (%) | Upscale+T=0.3 (%) |
|----------|-------------:|------------:|----------:|------------------:|
| M | 52.0 | 52.0 | 63.0 | 63.0 |
| N | 38.0 | 37.0 | 54.0 | 49.0 |
| SPACE | 53.0 | 44.0 | 64.0 | 56.0 |
| DELETE | 61.0 | 57.0 | 63.0 | 64.0 |
| P | 62.0 | 59.0 | 66.0 | 62.0 |
| X | 68.0 | 67.0 | 69.0 | 69.0 |
| C | 70.0 | 70.0 | 74.0 | 74.0 |
| Q | 71.0 | 67.0 | 74.0 | 72.0 |
| Z | 74.0 | 75.0 | 76.0 | 75.0 |
| J | 86.0 | 80.0 | 91.0 | 85.0 |
| **OVERALL** | **63.5** | **60.8** | **69.4** | **66.9** |

## Diagnostic Conclusions

### Does Upscaling Help?
No. Contrary to expectations, upscaling the 200x200 images to 400x400 actually *decreased* overall detection rates on the difficult subset (from 63.5% down to 60.8%). MediaPipe's internal preprocessing and anchor generation likely performed worse on the artificially bicubic-upsampled artifacts.

### Does Threshold Adjustment Help?
Yes, significantly. Dropping the threshold to 0.3 improved the usable detection yield across almost all problematic classes, pushing the difficult subset average from 63.5% to 69.4% (+5.9 percentage points). Because we strictly filter out non-finite and degenerate coordinates, the false-positive risk of `0.3` is safely mitigated by our usability check.

### Resulting Landmark Validity
All accepted detections pass the structural finiteness and boundary checks. No garbage coordinates were recorded as successes.

### Problematic Classes Remaining
While significantly improved, classes like `N` (54%), `M` (63%), and `SPACE` (64%) still trail others. This is fundamentally due to ASL topology — the fingers are folded over the palm, providing almost no distinct contour for a 2D RGB model to track.

### Recommendation for Production Extraction
**Configuration:** THRESH03 (Lower Threshold Only)
Use the original image resolution combined with a `0.3` confidence threshold for maximum usable extraction yield before feeding to the MLP.

It is now safe to proceed to full extraction using this configuration and the 28-class architecture.
