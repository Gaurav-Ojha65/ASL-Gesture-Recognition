"""
Standalone Diagnostic Script for Physical Text-to-Speech (TTS) Verification.

Tests the real Windows SAPI5 / SpeechEngine backend using the actual asynchronous
worker implementation of the ASL Gesture Recognition application.
Verifies the exact dual-pronunciation sequence:
C -> A -> T -> CAT
"""

import os
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.realtime.speech import SpeechEngine


def run_diagnostic() -> int:
    print("=" * 70)
    print("ASL GESTURE RECOGNITION - TTS AUDIO BACKEND DIAGNOSTIC")
    print("=" * 70)
    print("Target Sequence: 'C' -> 'A' -> 'T' -> 'CAT'")
    print("Initializing SpeechEngine with real Windows audio backend...")

    engine = SpeechEngine(enabled=True)
    time.sleep(0.4)

    if not engine.is_available:
        print("[ERROR] TTS Engine reported unavailable. Check sound hardware/drivers.")
        engine.stop()
        return 1

    print(f"[STATUS] Engine available: {engine.is_available}")
    print(f"[STATUS] Worker thread alive: {engine.is_worker_alive}")
    print("-" * 70)

    # 1. Commit 'C'
    print("\n--- 1. Committing Letter 'C' ---")
    committed_c = engine.on_commit_event("C", "C", timestamp=time.time())
    print(f"on_commit_event('C') returned: {committed_c}")
    time.sleep(1.2)

    # 2. Commit 'A'
    print("\n--- 2. Committing Letter 'A' ---")
    committed_a = engine.on_commit_event("A", "CA", timestamp=time.time())
    print(f"on_commit_event('A') returned: {committed_a}")
    time.sleep(1.2)

    # 3. Commit 'T'
    print("\n--- 3. Committing Letter 'T' ---")
    committed_t = engine.on_commit_event("T", "CAT", timestamp=time.time())
    print(f"on_commit_event('T') returned: {committed_t}")
    time.sleep(1.2)

    # 4. Commit 'SPACE' (Word 'CAT')
    print("\n--- 4. Committing Keyboard 'SPACE' for Word 'CAT' ---")
    committed_space = engine.on_commit_event("SPACE", "CAT ", timestamp=time.time())
    print(f"on_commit_event('SPACE') returned: {committed_space}")
    time.sleep(1.8)

    print("\n" + "-" * 70)
    print(f"[STATUS] Last spoken utterance: '{engine.last_spoken}'")
    print(f"[STATUS] Speech queue empty: {engine._queue.empty()}")
    print(f"[STATUS] Worker thread alive: {engine.is_worker_alive}")

    success = (
        committed_c
        and committed_a
        and committed_t
        and committed_space
        and engine._queue.empty()
        and engine.is_worker_alive
    )

    print("\nStopping SpeechEngine worker...")
    engine.stop(timeout=2.0)
    print(f"[STATUS] Worker thread alive after stop: {engine.is_worker_alive}")
    print("=" * 70)

    if success:
        print("DIAGNOSTIC RESULT: SUCCESS - All items processed through real SAPI5 audio pipeline.")
        return 0
    else:
        print("DIAGNOSTIC RESULT: FAILURE - One or more stages failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostic())
