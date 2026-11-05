# ASL Full Production Extraction Independent Audit

## 1. Split Specification Verdict: PASS
The extraction was verified to strictly assign images 1-2100 to Train, 2101-2550 to Validation, and 2551-3000 to Internal Test for EVERY class.
Split violations found: 0

## 2. Split Table
| Class | Train Attempted | Train Success | Val Attempted | Val Success | Test Attempted | Test Success |
| ----- | --------------: | ------------: | ------------: | ----------: | -------------: | -----------: |
| A | 2100 | 1796 | 450 | 354 | 450 | 145 |
| B | 2100 | 1809 | 450 | 381 | 450 | 60 |
| C | 2100 | 1773 | 450 | 267 | 450 | 79 |
| D | 2100 | 2018 | 450 | 401 | 450 | 164 |
| E | 2100 | 1864 | 450 | 373 | 450 | 107 |
| F | 2100 | 2031 | 450 | 426 | 450 | 450 |
| G | 2100 | 1961 | 450 | 352 | 450 | 249 |
| H | 2100 | 2029 | 450 | 365 | 450 | 78 |
| I | 2100 | 2060 | 450 | 329 | 450 | 84 |
| J | 2100 | 2031 | 450 | 394 | 450 | 222 |
| K | 2100 | 2052 | 450 | 397 | 450 | 293 |
| L | 2100 | 1997 | 450 | 358 | 450 | 244 |
| M | 2100 | 1516 | 450 | 348 | 450 | 93 |
| N | 2100 | 1241 | 450 | 210 | 450 | 78 |
| O | 2100 | 1984 | 450 | 183 | 450 | 182 |
| P | 2100 | 1866 | 450 | 233 | 450 | 28 |
| Q | 2100 | 1931 | 450 | 190 | 450 | 108 |
| R | 2100 | 2012 | 450 | 379 | 450 | 226 |
| S | 2100 | 2025 | 450 | 350 | 450 | 265 |
| T | 2100 | 2016 | 450 | 316 | 450 | 71 |
| U | 2100 | 1961 | 450 | 418 | 450 | 178 |
| V | 2100 | 1926 | 450 | 430 | 450 | 240 |
| W | 2100 | 1927 | 450 | 353 | 450 | 239 |
| X | 2100 | 1905 | 450 | 286 | 450 | 46 |
| Y | 2100 | 2083 | 450 | 392 | 450 | 162 |
| Z | 2100 | 1999 | 450 | 311 | 450 | 119 |
| DELETE | 2100 | 1770 | 450 | 170 | 450 | 102 |
| SPACE | 2100 | 1676 | 450 | 220 | 450 | 87 |


## 3. Global Sorting Bug Check
Verified that split assignment was performed independently within each class. There is NO global sorting bug.

## 4. Source Coverage Verdict: PASS
- Expected sources: 84000
- Successful sources: 66844
- NOTHING records: 0
- Duplicate source paths: 0
- Source paths in multiple splits: 0

## 5. Class / Label Integrity Verdict: PASS
- Exactly 28 semantic labels.
- Internal indices map perfectly to 0-27.

## 6. Feature Integrity Verdict: PASS
- Length strictly 63.
- All values finite.
- Stored as float arrays.

## 7. Failure Accounting Verdict: PASS
- Successful: 66844
- Failed: 17156
- Total: 84000

## 8. Ultimate Question
**"Does the existing Step 5.7 dataset exactly follow the required 2,100/450/450 split INSIDE EACH OF THE 28 CLASSES?"**
**YES**. The split rule was perfectly enforced.
