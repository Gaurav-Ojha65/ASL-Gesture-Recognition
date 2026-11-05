"""
Text Buffer Module for Real-Time ASL Gesture Recognition Interface.

Maintains an accumulated text string resulting from committed gesture events.
Handles:
- Uppercase letters (A-Z)
- Word separator (SPACE) with leading and consecutive space prevention
- Character deletion (DELETE) with safe empty-buffer handling
- NOTHING / NO_HAND as safe no-op
"""

from typing import Optional, Union, Any

class TextBuffer:
    """
    Accumulator for recognized ASL gestures into readable text.

    Guarantees:
    - Uppercase letter commits append to the buffer.
    - SPACE never creates a leading space or consecutive spaces.
    - DELETE safely removes the trailing character without raising errors on empty buffer.
    - NOTHING or NO_HAND tokens are safe no-ops that leave the text untouched.
    - Configurable maximum length rejects further additions without silent truncation.
    """

    def __init__(self, max_length: Optional[int] = None):
        """
        Args:
            max_length: Optional positive integer defining maximum allowed characters.
                        If reached, new letter/space additions are rejected (returns False)
                        to prevent silent truncation until characters are deleted or cleared.
        """
        if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
            raise ValueError(f"max_length must be a positive integer or None, got: {max_length}")

        self.max_length = max_length
        self._buffer: list[str] = []

    def apply(self, token: Any) -> bool:
        """
        Applies a committed gesture token to the buffer.

        Args:
            token: String semantic label ('A'-'Z', 'SPACE', 'DELETE', 'NOTHING', 'NO_HAND')
                   or an object with a 'label' attribute (e.g., CommitEvent).

        Returns:
            bool: True if the buffer content was modified, False otherwise.

        Raises:
            ValueError: If token is an unrecognized or invalid semantic string.
        """
        if token is None:
            return False

        # Support CommitEvent-like objects
        if hasattr(token, 'label'):
            token_str = token.label
        else:
            token_str = str(token)

        token_clean = token_str.strip().upper()

        # 1. Handle NOTHING / NO_HAND (Safe no-op)
        if token_clean in ('NOTHING', 'NO_HAND', ''):
            return False

        # 2. Handle DELETE
        if token_clean in ('DELETE', 'DEL'):
            if self._buffer:
                self._buffer.pop()
                return True
            return False

        # Check max length before adding new characters
        if self.max_length is not None and len(self._buffer) >= self.max_length:
            return False

        # 3. Handle SPACE
        if token_clean == 'SPACE':
            # Rule: Do not insert leading space into empty buffer
            if not self._buffer:
                return False
            # Rule: Do not insert consecutive spaces
            if self._buffer[-1] == ' ':
                return False
            self._buffer.append(' ')
            return True

        # 4. Handle Letters A-Z
        if len(token_clean) == 1 and 'A' <= token_clean <= 'Z':
            self._buffer.append(token_clean)
            return True

        # 5. Unsupported label
        raise ValueError(f"Unsupported token for TextBuffer: '{token_str}'")

    def get_text(self) -> str:
        """Returns the accumulated text as a string."""
        return "".join(self._buffer)

    def clear(self) -> None:
        """Clears all text from the buffer."""
        self._buffer.clear()

    @property
    def character_count(self) -> int:
        """Returns the current number of characters in the buffer."""
        return len(self._buffer)

    def __len__(self) -> int:
        """Returns the current number of characters in the buffer."""
        return len(self._buffer)

    def __str__(self) -> str:
        return self.get_text()

    def __repr__(self) -> str:
        return f"TextBuffer(text='{self.get_text()}', length={len(self._buffer)}, max_length={self.max_length})"
