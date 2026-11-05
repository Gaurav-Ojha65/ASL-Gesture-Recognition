# ASL Landmark Normalization Report

## Normalization Algorithm
1. **Validation**: Check for exactly 21 landmarks with finite (x, y, z) coordinates.
2. **Translation Invariance**: The wrist landmark (index 0) is set as the origin `(0,0,0)` by subtracting its coordinates from all landmarks.
3. **Scale Invariance**: The 3D euclidean distance between the wrist and the middle finger MCP (index 9) is used as the reference scale factor. All coordinates are divided by this scale. Zero or near-zero scales are explicitly checked and safely rejected.
4. **Handedness Canonicalization (Right-Handed Space)**: MediaPipe's 'Left' and 'Right' classifications refer to physical handedness. In our front-facing / 3rd-person perspective ASL dataset, all representations are transformed into a canonical Right-hand space. If MediaPipe detects a 'Left' hand, the X-coordinates of all landmarks are inverted (`x = -x`) relative to the wrist origin.
5. **Flattening**: The normalized 3D coordinates are flattened into a `float32` vector of length 63.

## Architectural Note
`NOTHING` (NO_HAND) remains a system-level state. This normalization pipeline strictly expects valid hands and does not zero-fill or fabricate data for missing hands. The subsequent MLP will classify 28 hand-bearing classes.

## Invariance Unit Tests
All 9 core unit tests passed successfully:
- Output shape is exactly (63,).
- Output dtype is float32.
- All values are finite.
- Translation invariance (translating hand yields same features).
- Scale invariance (scaling hand yields same features).
- Zero-scale handling (safely rejects degenerate scale).
- Wrist becomes the origin after normalization `(0,0,0)`.
- Handedness behavior (Left-handed mirrored samples yield identical right-handed canonicalized features).
- No accidental mutation of original array.

## Pilot Data Analysis Results

### Processing Summary
- **Total pilot detections processed:** 2099
- **Valid normalized samples:** 2099
- **Invalid normalized samples:** 0

### Numerical Statistics
- **Feature shape:** (2099, 63)
- **Dtype:** float32
- **Contains NaN/Inf:** False
- **Global Min:** -3.205479
- **Global Max:** 4.506077
- **Global Mean:** -0.361206
- **Global Std:** 0.596675

### Original Handedness Breakdown
- **Right:** 1968
- **Left:** 131

### Class-wise Valid Normalized Counts
- **A:** 77
- **B:** 74
- **C:** 69
- **D:** 81
- **DELETE:** 59
- **E:** 78
- **F:** 95
- **G:** 79
- **H:** 77
- **I:** 79
- **J:** 86
- **K:** 88
- **L:** 84
- **M:** 52
- **N:** 39
- **O:** 77
- **P:** 63
- **Q:** 70
- **R:** 84
- **S:** 86
- **SPACE:** 51
- **T:** 77
- **U:** 86
- **V:** 80
- **W:** 80
- **X:** 68
- **Y:** 86
- **Z:** 74

*Conclusion: The normalization pipeline is numerically stable, fully canonicalizes spatial variables, and safely rejects invalid data. It is ready for the full dataset extraction.*