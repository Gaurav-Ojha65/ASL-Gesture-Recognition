# ASL Full Production Extraction Report

## 1. Overview
- **Configuration:** MediaPipe `0.3` Threshold, Strict Contiguous Block Split.
- **Source Dataset:** `data/raw/asl_alphabet/asl_alphabet_train/`
- **Number of Classes:** 28 (Hand-Bearing Only)
- **NOTHING processed?** No. (Explicitly skipped).

## 2. Extraction Statistics
- **Expected Source Images (Attempted):** 84000
- **Successful Extractions:** 66844
- **Failed Extractions:** 17156
- **Overall Success Rate:** 79.58%

### Split Statistics
- **Train:** 53259
- **Validation:** 9186
- **Internal Test:** 4399

### Class-Wise Statistics
| Class | Attempted | Success | Fail | Success Rate |
|---|---|---|---|---|
| A | 3000 | 2295 | 705 | 76.5% |
| B | 3000 | 2250 | 750 | 75.0% |
| C | 3000 | 2119 | 881 | 70.6% |
| D | 3000 | 2583 | 417 | 86.1% |
| E | 3000 | 2344 | 656 | 78.1% |
| F | 3000 | 2907 | 93 | 96.9% |
| G | 3000 | 2562 | 438 | 85.4% |
| H | 3000 | 2472 | 528 | 82.4% |
| I | 3000 | 2473 | 527 | 82.4% |
| J | 3000 | 2647 | 353 | 88.2% |
| K | 3000 | 2742 | 258 | 91.4% |
| L | 3000 | 2599 | 401 | 86.6% |
| M | 3000 | 1957 | 1043 | 65.2% |
| N | 3000 | 1529 | 1471 | 51.0% |
| O | 3000 | 2349 | 651 | 78.3% |
| P | 3000 | 2127 | 873 | 70.9% |
| Q | 3000 | 2229 | 771 | 74.3% |
| R | 3000 | 2617 | 383 | 87.2% |
| S | 3000 | 2640 | 360 | 88.0% |
| T | 3000 | 2403 | 597 | 80.1% |
| U | 3000 | 2557 | 443 | 85.2% |
| V | 3000 | 2596 | 404 | 86.5% |
| W | 3000 | 2519 | 481 | 84.0% |
| X | 3000 | 2237 | 763 | 74.6% |
| Y | 3000 | 2637 | 363 | 87.9% |
| Z | 3000 | 2429 | 571 | 81.0% |
| DELETE | 3000 | 2042 | 958 | 68.1% |
| SPACE | 3000 | 1983 | 1017 | 66.1% |

### Failure Categories
- **No hand detected:** 17156

## 3. Storage & Performance
- **Parquet Schema:** FixedSizeList[Float32, 63] (PyArrow)
- **Total Parquet Disk Size:** 24.65 MB
- **Runtime:** 5512.3 seconds
- **Throughput:** 15.2 images/sec

## 4. Final Verification
All strict validation checks against the final generated Parquet dataset were evaluated:
- Total row count matches extraction count: PASS
- No duplicate source paths: PASS
- No source path appears in multiple splits: PASS
- Only the 28 allowed semantic labels occur: PASS
- Internal class indices exactly 0-27: PASS
- Feature vectors have exactly 63 elements: PASS
- Features are float: PASS
- No NaN or Inf exists: PASS
- Split values strictly known: PASS
- Safe single-file exclusivity: PASS

*Conclusion: The dataset is mathematically pristine, normalized, safely persisted, and ready for PyTorch Model Training.*
