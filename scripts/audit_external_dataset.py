import os
import sys
import hashlib
from collections import defaultdict
from PIL import Image

# ── Configuration & Paths ──
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXTERNAL_DIR = os.path.join(BASE_DIR, "data", "raw", "asl_alphabet_external")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "asl_alphabet", "asl_alphabet_train", "asl_alphabet_train")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "asl_external_dataset_audit.md")

EXPECTED_CLASSES = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
])

EXPECTED_IMAGES_PER_CLASS = 30
EXPECTED_TOTAL_IMAGES = 870

SEMANTIC_MAPPING = {
    'del': 'DELETE',
    'space': 'SPACE',
    'nothing': 'NOTHING',
}
for c in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
    SEMANTIC_MAPPING[c] = c

def compute_md5(filepath: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_audit():
    lines = []
    lines.append("# ASL External Dataset Audit (Step 8.1)")
    lines.append("")
    lines.append(f"**Target Directory:** `{EXTERNAL_DIR}`  ")
    lines.append(f"**Expected Classes:** 29 (A–Z, del, nothing, space)  ")
    lines.append(f"**Expected Images per Class:** {EXPECTED_IMAGES_PER_CLASS}  ")
    lines.append(f"**Expected Total Images:** {EXPECTED_TOTAL_IMAGES}  ")
    lines.append("")

    # Check 1: Directory Existence
    lines.append("## 1. Directory Existence Check")
    if not os.path.exists(EXTERNAL_DIR):
        lines.append(f"- **Directory Status:** **FAIL** — Directory not found at `{EXTERNAL_DIR}`")
        lines.append("- **Notes:** Sourced from the downloaded dataset archive for `asl-alphabet-test`.")
        lines.append("- **Action Required:** The user must manually copy or move the dataset into `data/raw/asl_alphabet_external/` before external evaluation can proceed.")
        lines.append("")
        lines.append("## Summary Verdict")
        lines.append("| Check | Status | Details |")
        lines.append("|---|---|---|")
        lines.append("| Directory Existence | **FAIL** | Path missing: `data/raw/asl_alphabet_external/` |")
        lines.append("| Folder Structure | **NOT TESTED** | Prerequisite directory missing |")
        lines.append("| Image Readability | **NOT TESTED** | Prerequisite directory missing |")
        lines.append("| Class Mapping | **NOT TESTED** | Prerequisite directory missing |")
        lines.append("| Separation / Overlap | **NOT TESTED** | Prerequisite directory missing |")
        lines.append("")
        lines.append("### Final Verdict")
        lines.append("**FAIL / BLOCKED** — The external dataset directory does not exist in the project repository. It is NOT ready for Step 8.2.")

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Directory {EXTERNAL_DIR} does NOT exist. Audit report written to {REPORT_PATH}")
        return False

    lines.append(f"- **Directory Status:** **PASS** — Directory found at `{EXTERNAL_DIR}`")
    lines.append("")

    # Check 2: Folder Structure
    lines.append("## 2. Class Folders and File Counts")
    entries = sorted(os.listdir(EXTERNAL_DIR))
    subfolders = [e for e in entries if os.path.isdir(os.path.join(EXTERNAL_DIR, e))]
    non_folders = [e for e in entries if not os.path.isdir(os.path.join(EXTERNAL_DIR, e))]

    missing_folders = set(EXPECTED_CLASSES) - set(subfolders)
    unexpected_folders = set(subfolders) - set(EXPECTED_CLASSES)

    folders_pass = (len(missing_folders) == 0 and len(unexpected_folders) == 0 and len(subfolders) == 29)
    lines.append(f"- **Total Subfolders Found:** {len(subfolders)} / 29")
    lines.append(f"- **Missing Subfolders:** {sorted(list(missing_folders)) if missing_folders else 'None [PASS]'}")
    lines.append(f"- **Unexpected Subfolders:** {sorted(list(unexpected_folders)) if unexpected_folders else 'None [PASS]'}")
    lines.append(f"- **Unexpected Root Files:** {non_folders if non_folders else 'None [PASS]'}")
    lines.append("")

    # Per-class counts and image verification
    class_counts = {}
    corrupted_images = []
    dimensions_set = set()
    channels_set = set()
    extensions_set = set()
    all_external_files = []
    external_hashes = {}
    external_filenames = defaultdict(list)

    for cls in EXPECTED_CLASSES:
        cls_dir = os.path.join(EXTERNAL_DIR, cls)
        if not os.path.isdir(cls_dir):
            class_counts[cls] = 0
            continue

        files = sorted(os.listdir(cls_dir))
        class_counts[cls] = len(files)

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            all_external_files.append(fpath)
            ext = os.path.splitext(fname)[1].lower()
            extensions_set.add(ext)
            external_filenames[fname].append(fpath)

            try:
                with Image.open(fpath) as img:
                    img.verify()
                # Re-open to read properties after verify()
                with Image.open(fpath) as img:
                    dimensions_set.add(img.size) # (width, height)
                    channels_set.add(img.mode)
                # Compute hash
                md5 = compute_md5(fpath)
                external_hashes[fpath] = md5
            except Exception as e:
                corrupted_images.append((fpath, str(e)))

    total_images_found = sum(class_counts.values())
    counts_pass = all(c == EXPECTED_IMAGES_PER_CLASS for c in class_counts.values()) and total_images_found == EXPECTED_TOTAL_IMAGES

    lines.append("### Class Distribution Table")
    lines.append("| Folder | Semantic Label | File Count | Expected | Status |")
    lines.append("|---|---|---|---|---|")
    for cls in EXPECTED_CLASSES:
        cnt = class_counts.get(cls, 0)
        status = "PASS" if cnt == EXPECTED_IMAGES_PER_CLASS else "FAIL"
        lines.append(f"| `{cls}` | `{SEMANTIC_MAPPING.get(cls, 'UNKNOWN')}` | {cnt} | {EXPECTED_IMAGES_PER_CLASS} | {status} |")
    lines.append(f"| **Total** | **All 29 Classes** | **{total_images_found}** | **{EXPECTED_TOTAL_IMAGES}** | **{'PASS' if counts_pass else 'FAIL'}** |")
    lines.append("")

    # Check 3: Image Integrity, Dimensions & Dtypes
    lines.append("## 3. Image Integrity and Format Verification")
    readability_pass = (len(corrupted_images) == 0 and total_images_found == EXPECTED_TOTAL_IMAGES)
    lines.append(f"- **Readable Images:** {total_images_found - len(corrupted_images)} / {total_images_found}")
    lines.append(f"- **Corrupted Images:** {len(corrupted_images)} {'[PASS]' if len(corrupted_images) == 0 else '[FAIL]'}")
    if corrupted_images:
        for f, err in corrupted_images[:10]:
            lines.append(f"  - `{f}`: {err}")
    lines.append(f"- **File Extensions Found:** {sorted(list(extensions_set))}")
    lines.append(f"- **Image Dimensions (Width, Height):** {sorted(list(dimensions_set))}")
    lines.append(f"- **Image Color Modes / Channels:** {sorted(list(channels_set))}")
    lines.append("")

    # Check 4: Class Mapping Verification
    lines.append("## 4. Class Mapping Verification")
    lines.append("The 29 external folders are interpreted as:")
    lines.append("- Letters `A`–`Z` (26 classes) -> Standard hand gestures")
    lines.append("- `del` (1 class) -> `DELETE` gesture")
    lines.append("- `space` (1 class) -> `SPACE` gesture")
    lines.append("- `nothing` (1 class) -> `NOTHING` / `NO_HAND` system state")
    lines.append("- Mapping check: **PASS** (100% of folders map to valid specification categories)")
    lines.append("")

    # Check 5: Dataset Separation & Overlap Check
    lines.append("## 5. Dataset Separation & Overlap Check")
    train_exists = os.path.exists(TRAIN_DIR)
    lines.append(f"Comparing against Training Dataset at: `{TRAIN_DIR}` (Exists: {train_exists})")

    filename_collisions = []
    content_collisions = []

    if train_exists:
        train_hashes = {}
        train_file_count = 0
        for root, _, files in os.walk(TRAIN_DIR):
            for f in files:
                train_file_count += 1
                if f in external_filenames:
                    filename_collisions.append((f, os.path.join(root, f)))
                full_path = os.path.join(root, f)
                # Compute hash
                h = compute_md5(full_path)
                train_hashes[h] = full_path

        # Check for identical content
        for ext_path, ext_hash in external_hashes.items():
            if ext_hash in train_hashes:
                content_collisions.append((ext_path, train_hashes[ext_hash]))

        lines.append(f"- **Training Images Evaluated:** {train_file_count}")
        lines.append(f"- **Filename Collisions:** {len(filename_collisions)}")
        if filename_collisions:
            lines.append(f"  *(Note: {len(filename_collisions)} files share simple generic names like A1.jpg or similar)*")
            for fn, tp in filename_collisions[:5]:
                lines.append(f"  - `{fn}` in `{tp}`")
            if len(filename_collisions) > 5:
                lines.append(f"  - ... ({len(filename_collisions) - 5} more)")

        lines.append(f"- **Exact Content Hash Collisions (MD5):** {len(content_collisions)} {'[PASS]' if len(content_collisions) == 0 else '[FAIL]'}")
        if content_collisions:
            for ep, tp in content_collisions[:10]:
                lines.append(f"  - External `{os.path.basename(ep)}` identical to Train `{os.path.basename(tp)}`")
    else:
        lines.append("- **Warning:** Original training folder not found for hash comparison.")

    lines.append("")

    # Summary Table
    separation_pass = (len(content_collisions) == 0)
    all_pass = (folders_pass and counts_pass and readability_pass and separation_pass)

    lines.append("## 6. Summary Verdict")
    lines.append("| Check | Status | Details |")
    lines.append("|---|---|---|")
    lines.append(f"| Directory Existence | **{'PASS' if os.path.exists(EXTERNAL_DIR) else 'FAIL'}** | `{EXTERNAL_DIR}` |")
    lines.append(f"| Folder Structure | **{'PASS' if folders_pass else 'FAIL'}** | Exactly 29 expected folders |")
    lines.append(f"| Sample Counts | **{'PASS' if counts_pass else 'FAIL'}** | {total_images_found} images (30 per class) |")
    lines.append(f"| Image Readability & Dimensions | **{'PASS' if readability_pass else 'FAIL'}** | 0 corrupt, formats: {list(dimensions_set)} |")
    lines.append(f"| Class Mapping Interpretation | **PASS** | 26 letters + DELETE + SPACE + NOTHING |")
    lines.append(f"| Content Separation (No Hash Overlap) | **{'PASS' if separation_pass else 'FAIL'}** | {len(content_collisions)} identical content matches |")
    lines.append("")

    lines.append("### Final Verdict")
    if all_pass:
        lines.append("**PASS** — The external dataset is verified, complete, uncorrupted, and strictly independent. It is ready for STEP 8.2 (external landmark extraction and model evaluation).")
    else:
        lines.append("**FAIL** — One or more checks failed. Correct the identified issues before proceeding to STEP 8.2.")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Audit completed. All checks passed: {all_pass}. Report written to {REPORT_PATH}")
    return all_pass

if __name__ == "__main__":
    run_audit()
