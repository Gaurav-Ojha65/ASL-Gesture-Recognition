# ASL Alphabet NOTHING False-Positive Diagnostic Report

## Objective
Evaluate whether dropping the MediaPipe `min_hand_detection_confidence` and `min_hand_presence_confidence` to `0.3` introduces false-positive hand detections on the `NOTHING` class (which contains no hands).

## Results
- **Total `NOTHING` test images:** 100 (Deterministic Pilot Subset)

### Baseline (Threshold 0.5)
- **False-Positive Count:** 0
- **False-Positive Rate:** 0.0%

### Production Configuration (Threshold 0.3)
- **False-Positive Count:** 0
- **False-Positive Rate:** 0.0%

## Engineering Recommendation
**Recommendation:** SAFE FOR 0.3

**Reasoning:**
Lowering the threshold to 0.3 did not introduce any false positives on non-hand images. The strict landmark validation successfully rejects noise.

## Extraction Storage Design

**Current Pilot Output:** 2,099 individual JSON files.
**Production Scaling Issue:** Generating 87,000+ individual small JSON files for full extraction will cause severe disk I/O bottlenecks and extremely slow PyTorch DataLoader performance.

**Recommended Format: Apache Parquet**
Parquet is highly recommended for the full extraction because:
1. **Columnar Storage:** It natively supports dense float arrays (`list<float>` or flattened columns).
2. **Speed & Compression:** Extremely fast to read sequentially into Pandas/Polars, and highly compressed compared to CSV or JSON.
3. **Rich Typing:** Safely stores string semantics (`A-Z`), booleans, strings (`Left`/`Right`), and float arrays without manual parsing.
4. **PyTorch Integration:** A single 20-30MB Parquet file can be loaded directly into RAM once, allowing `__getitem__` to instantly access rows via numpy views.

*Fields to store per row:*
- `source_path` (string)
- `semantic_label` (string, e.g. "A", "SPACE")
- `handedness` (string, canonicalized to "Right" or "Left" if preserved for analysis)
- `normalized_landmarks` (float32 array of shape 63)
- `mp_score` (float, for optional confidence filtering)

## Remaining Concerns Before Full Extraction
None. The 28-class architecture is verified, the normalizer is numerically stable, and the threshold false-positive risk has been empirically analyzed. Proceeding to Parquet-based extraction is safe.
