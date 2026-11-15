"""
Asynchronous Non-Blocking Text-to-Speech (TTS) Module for ASL Gesture Recognition.

Provides offline, thread-safe speech synthesis using pyttsx3 (SAPI5 on Windows).
Key Features:
- Asynchronous background worker thread with thread-safe queue.
- Engine initialization isolated within worker thread (ensures COM apartment stability on Windows).
- Both individual letter pronunciation on commit and word-level pronunciation on SPACE / pause.
- Duplicate speech suppression and whitespace filtering.
- Non-blocking execution ensuring 30+ FPS camera capture loop is never interrupted.
- Keyboard toggling (Speech: ON / OFF) and graceful shutdown without hanging.
- Safe fallback if audio devices or drivers are unavailable.
"""

import sys
import queue
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class SpeechEngine:
    """
    Asynchronous Speech Engine that consumes speech requests from a thread-safe queue.
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 160,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
        pause_speech_delay: float = 2.5,
        engine_factory: Optional[Callable[[], Any]] = None,
        letter_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            enabled: Initial speech state (default: True).
            rate: Speech rate in words per minute (default: 160).
            volume: Speech volume from 0.0 to 1.0 (default: 1.0).
            voice_id: Specific voice ID or None for default system voice.
            pause_speech_delay: Inactivity pause duration (seconds) before speaking unspaced word.
            engine_factory: Factory returning a pyttsx3-compatible engine (useful for unit testing/mocking).
            letter_map: Optional dictionary mapping letters A-Z to specific spoken letter names.
        """
        self._enabled = enabled
        self._rate = rate
        self._volume = max(0.0, min(1.0, volume))
        self._voice_id = voice_id
        self._pause_speech_delay = pause_speech_delay
        self._engine_factory = engine_factory
        self._letter_map = letter_map or {c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

        # Threading and Queue
        self._queue: queue.Queue = queue.Queue(maxsize=30)
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # Engine state
        self._is_available: bool = False
        self._is_speaking: bool = False
        self._last_spoken: str = ""
        self._last_spoken_time: float = 0.0

        # Word tracking for pause speech & duplicate prevention
        self._pending_word: str = ""
        self._pending_word_time: float = 0.0
        self._pending_word_spoken: bool = False

        # Start background worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ASL_SpeechWorker",
            daemon=True
        )
        self._worker_thread.start()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def is_worker_alive(self) -> bool:
        return self._worker_thread.is_alive() if hasattr(self, "_worker_thread") else False

    @property
    def last_spoken(self) -> str:
        with self._lock:
            return self._last_spoken

    def toggle(self) -> bool:
        """Toggles speech enabled/disabled. Returns new state."""
        with self._lock:
            self._enabled = not self._enabled
            # Clear pending queue on disable
            if not self._enabled:
                self._clear_queue()
                self._pending_word = ""
                self._pending_word_spoken = False
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Explicitly sets speech state."""
        with self._lock:
            self._enabled = enabled
            if not self._enabled:
                self._clear_queue()
                self._pending_word = ""
                self._pending_word_spoken = False

    def _clear_queue(self) -> None:
        """Flushes all queued speech items."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break

    # ── Speech Triggers ───────────────────────────────────────────────────────

    def speak(self, text: str, force: bool = False) -> bool:
        """
        Queues a piece of text to be spoken.

        Args:
            text: Text to speak.
            force: If True, bypasses duplicate suppression check.

        Returns:
            bool: True if queued, False if rejected (disabled, empty, duplicate, or queue full).
        """
        if not self._enabled:
            return False

        clean_text = text.strip()
        if not clean_text:
            return False

        with self._lock:
            # Duplicate suppression check (within 4 seconds)
            now = time.time()
            if not force and clean_text == self._last_spoken and (now - self._last_spoken_time) < 4.0:
                return False

            try:
                self._queue.put_nowait(clean_text)
                # Mark as enqueued to prevent immediate duplicate spam before worker thread pops it
                self._last_spoken = clean_text
                self._last_spoken_time = now
                alive = self._worker_thread.is_alive() if hasattr(self, "_worker_thread") else False
                print(f"[TTS ENQUEUE] '{clean_text}' (worker_alive={alive})")
                return True
            except queue.Full:
                print(f"[TTS ERROR] Speech queue full. Dropped utterance: '{clean_text}'", file=sys.stderr)
                logger.warning("Speech queue full. Dropping utterance: '%s'", clean_text)
                return False

    def on_commit_event(
        self,
        token: str,
        full_buffer_text: str,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Called when a token is committed to the text buffer.
        - If token is a letter (A-Z), enqueues the individual letter name immediately
          and updates the pending word.
        - If token is 'SPACE', speaks the complete word (unless already spoken via pause).
        - If token is 'DELETE', updates pending word without speaking.
        - If token is 'CLEAR', resets pending word state without speaking.
        """
        token_upper = token.strip().upper()
        event_time = timestamp if timestamp is not None else time.time()

        if token_upper == "SPACE":
            # Extract last word before the space
            words = full_buffer_text.strip().split()
            if words:
                last_word = words[-1]
                with self._lock:
                    already_spoken = (self._pending_word == last_word and self._pending_word_spoken)
                    self._pending_word = ""
                    self._pending_word_time = 0.0
                    self._pending_word_spoken = False
                if already_spoken:
                    return False
                return self.speak(last_word, force=True)
            return False

        elif token_upper in ("DELETE", "DEL"):
            with self._lock:
                words = full_buffer_text.strip().split()
                self._pending_word = words[-1] if words else ""
                self._pending_word_time = event_time
                self._pending_word_spoken = False
            return False

        elif token_upper == "CLEAR":
            with self._lock:
                self._pending_word = ""
                self._pending_word_time = 0.0
                self._pending_word_spoken = False
            return False

        elif len(token_upper) == 1 and "A" <= token_upper <= "Z":
            spoken_letter = self._letter_map.get(token_upper, token_upper)
            with self._lock:
                words = full_buffer_text.strip().split()
                self._pending_word = words[-1] if words else token_upper
                self._pending_word_time = event_time
                self._pending_word_spoken = False
            return self.speak(spoken_letter, force=True)

        return False

    def update_pause_check(self, current_time: float) -> bool:
        """
        Checks if an unspaced word has been idle for >= pause_speech_delay seconds.
        If so, speaks the completed word.
        """
        word_to_speak = ""
        with self._lock:
            if not self._enabled or not self._pending_word or self._pending_word_spoken:
                return False

            if (current_time - self._pending_word_time) >= self._pause_speech_delay:
                word_to_speak = self._pending_word
                self._pending_word_spoken = True

        if word_to_speak:
            return self.speak(word_to_speak, force=True)
        return False

    # ── Background Worker Loop ────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Dedicated thread that initializes the TTS engine and processes speech queue.
        Initializes COM apartment and backend directly on this worker thread.
        Prefers Windows SAPI5 (win32com) for reliable, non-blocking synchronous execution;
        falls back gracefully to pyttsx3 or custom engine factory.
        """
        engine = None
        backend_name = "None"
        pythoncom_initialized = False

        try:
            if self._engine_factory is not None:
                engine = self._engine_factory()
                backend_name = "Mock/Custom Factory"
                self._is_available = True
            else:
                # 1. Prefer native Windows SAPI5 via win32com (direct, synchronous within thread, zero event-loop hangs)
                try:
                    import pythoncom
                    import win32com.client
                    pythoncom.CoInitialize()
                    pythoncom_initialized = True
                    engine = win32com.client.Dispatch("SAPI.SpVoice")
                    # Set properties
                    if self._volume is not None:
                        engine.Volume = int(self._volume * 100)
                    if self._rate is not None:
                        # Convert WPM to SAPI rate scale (-10 to +10, 0 is ~160 wpm)
                        engine.Rate = max(-10, min(10, int((self._rate - 160) / 20)))
                    backend_name = "Windows SAPI5 (win32com)"
                    self._is_available = True
                except Exception as e_sapi:
                    logger.debug("win32com SAPI init failed (%s), falling back to pyttsx3", e_sapi)
                    # 2. Fallback to pyttsx3
                    import pyttsx3
                    engine = pyttsx3.init()
                    if self._rate is not None:
                        engine.setProperty("rate", self._rate)
                    if self._volume is not None:
                        engine.setProperty("volume", self._volume)
                    if self._voice_id is not None:
                        engine.setProperty("voice", self._voice_id)
                    backend_name = "pyttsx3"
                    self._is_available = True

            print(f"[TTS READY] Speech worker initialized successfully ({backend_name})")
        except Exception as e:
            print(f"[TTS ERROR] Failed to initialize text-to-speech engine: {e}", file=sys.stderr)
            logger.error("Failed to initialize text-to-speech engine: %s", e)
            self._is_available = False

        # Main worker message loop
        while not self._shutdown_event.is_set():
            try:
                # Wait for utterance with timeout to allow shutdown polling
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None or self._shutdown_event.is_set():
                # Shutdown sentinel received
                break

            if not self._enabled or not self._is_available or engine is None:
                self._queue.task_done()
                continue

            # Process utterance with full diagnostic logging
            print(f"[TTS DEQUEUE] '{item}'")
            print(f"[TTS START] '{item}'")
            try:
                with self._lock:
                    self._is_speaking = True
                    self._last_spoken = item
                    self._last_spoken_time = time.time()

                if hasattr(engine, "Speak"):
                    engine.Speak(item, 0)  # 0 = SVSFDefault (synchronous inside worker thread)
                elif hasattr(engine, "say") and hasattr(engine, "runAndWait"):
                    engine.say(item)
                    engine.runAndWait()

                print(f"[TTS COMPLETE] '{item}'")
            except Exception as e:
                print(f"[TTS ERROR] Utterance '{item}' failed: {e}", file=sys.stderr)
                logger.error("Speech synthesis error on utterance '%s': %s", item, e)
            finally:
                with self._lock:
                    self._is_speaking = False
                self._queue.task_done()

        # Clean shutdown of engine
        if engine is not None and hasattr(engine, "stop"):
            try:
                engine.stop()
            except Exception:
                pass

        if pythoncom_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

        print("[TTS SHUTDOWN] Speech worker thread exited cleanly.")

    def stop(self, timeout: float = 2.0) -> None:
        """
        Gracefully signals worker thread to shut down and joins it.
        """
        self._shutdown_event.set()
        try:
            self._queue.put_nowait(None)  # Sentinel to wake up queue.get()
        except queue.Full:
            pass

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
