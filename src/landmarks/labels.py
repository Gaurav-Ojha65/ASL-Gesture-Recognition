"""
Authoritative label mapping for ASL Alphabet.

Maps 28 hand-bearing semantic classes to contiguous internal integer indices (0-27).
Excludes the 'NOTHING' class, which is handled as a system-level state (NO_HAND).
"""

# The 28 valid hand-bearing classes
SEMANTIC_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'DELETE', 'SPACE'
]

# Ensure exactly 28 classes
assert len(SEMANTIC_CLASSES) == 28

# Create bidirectional mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SEMANTIC_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(SEMANTIC_CLASSES)}

def get_class_index(semantic_label: str) -> int:
    """Returns the internal integer index (0-27) for a given semantic label."""
    if semantic_label not in CLASS_TO_IDX:
        raise ValueError(f"Unknown or invalid semantic label: '{semantic_label}'. "
                         "Note: 'NOTHING' is not a valid landmark class.")
    return CLASS_TO_IDX[semantic_label]

def get_semantic_label(class_index: int) -> str:
    """Returns the semantic label for a given internal integer index."""
    if class_index not in IDX_TO_CLASS:
        raise ValueError(f"Unknown internal class index: {class_index}")
    return IDX_TO_CLASS[class_index]
