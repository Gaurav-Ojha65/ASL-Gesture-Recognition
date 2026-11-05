import pytest
from src.realtime.text_buffer import TextBuffer
from src.realtime.gesture_state_machine import CommitEvent

def test_append_individual_letters():
    """1. Appending individual letters."""
    buf = TextBuffer()
    assert buf.apply('A') is True
    assert buf.get_text() == "A"
    assert len(buf) == 1
    assert buf.character_count == 1

    assert buf.apply('b') is True # lowercase normalized to uppercase
    assert buf.get_text() == "AB"
    assert len(buf) == 2

def test_build_word_asl():
    """2. Building a word such as ASL."""
    buf = TextBuffer()
    for letter in ['A', 'S', 'L']:
        buf.apply(letter)
    assert buf.get_text() == "ASL"
    assert len(buf) == 3

def test_space_between_words():
    """3. Adding a SPACE between words."""
    buf = TextBuffer()
    buf.apply('H')
    buf.apply('I')
    assert buf.apply('SPACE') is True
    buf.apply('Y')
    buf.apply('O')
    buf.apply('U')
    assert buf.get_text() == "HI YOU"

def test_prevent_leading_space():
    """4. Preventing leading spaces."""
    buf = TextBuffer()
    assert buf.apply('SPACE') is False
    assert buf.get_text() == ""
    assert len(buf) == 0

def test_prevent_consecutive_spaces():
    """5. Preventing consecutive spaces."""
    buf = TextBuffer()
    buf.apply('A')
    assert buf.apply('SPACE') is True
    assert buf.get_text() == "A "
    assert buf.apply('SPACE') is False
    assert buf.apply('SPACE') is False
    assert buf.get_text() == "A "

def test_delete_one_character():
    """6. Deleting one character."""
    buf = TextBuffer()
    buf.apply('A')
    buf.apply('B')
    assert buf.get_text() == "AB"
    assert buf.apply('DELETE') is True
    assert buf.get_text() == "A"
    assert buf.apply('DEL') is True
    assert buf.get_text() == ""

def test_delete_empty_buffer():
    """7. Deleting from an empty buffer is safe and does not raise an exception."""
    buf = TextBuffer()
    assert buf.apply('DELETE') is False
    assert buf.apply('DEL') is False
    assert buf.get_text() == ""
    assert len(buf) == 0

def test_nothing_noop():
    """8. Applying NOTHING without changing text."""
    buf = TextBuffer()
    buf.apply('A')
    assert buf.apply('NOTHING') is False
    assert buf.apply('NO_HAND') is False
    assert buf.apply(None) is False
    assert buf.get_text() == "A"

def test_clear_buffer():
    """9. Clearing the buffer."""
    buf = TextBuffer()
    buf.apply('A')
    buf.apply('B')
    buf.apply('C')
    assert buf.get_text() == "ABC"
    buf.clear()
    assert buf.get_text() == ""
    assert len(buf) == 0

def test_representative_sequence():
    """10. Applying sequence: A, S, L, SPACE, A, DELETE -> expected 'ASL '."""
    buf = TextBuffer()
    sequence = ['A', 'S', 'L', 'SPACE', 'A', 'DELETE']
    for token in sequence:
        buf.apply(token)
    assert buf.get_text() == "ASL "

def test_commit_event_compatibility():
    """Test compatibility with CommitEvent instances."""
    buf = TextBuffer()
    event_a = CommitEvent(label='A', timestamp=1.0, confidence=0.95, dwell_time=0.6)
    event_space = CommitEvent(label='SPACE', timestamp=2.0, confidence=0.92, dwell_time=0.6)
    event_b = CommitEvent(label='B', timestamp=3.0, confidence=0.88, dwell_time=0.6)

    assert buf.apply(event_a) is True
    assert buf.apply(event_space) is True
    assert buf.apply(event_b) is True
    assert buf.get_text() == "A B"

def test_max_length_configuration():
    """Test configurable max buffer length."""
    buf = TextBuffer(max_length=3)
    assert buf.apply('A') is True
    assert buf.apply('B') is True
    assert buf.apply('C') is True
    # At max length: additions rejected, no silent truncation
    assert buf.apply('D') is False
    assert buf.get_text() == "ABC"
    # Deleting makes room
    assert buf.apply('DELETE') is True
    assert buf.get_text() == "AB"
    assert buf.apply('Z') is True
    assert buf.get_text() == "ABZ"

def test_invalid_tokens_raise_error():
    """Test that unsupported tokens raise ValueError."""
    buf = TextBuffer()
    with pytest.raises(ValueError):
        buf.apply('UNKNOWN_TOKEN')
    with pytest.raises(ValueError):
        buf.apply('123')
