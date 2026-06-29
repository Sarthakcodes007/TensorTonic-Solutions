import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    # Handle empty input
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    # Determine maximum length if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    result = []

    for seq in seqs:
        # Truncate if longer than max_len
        seq = seq[:max_len]

        # Pad if shorter than max_len
        padding = max_len - len(seq)
        result.append(seq + [pad_value] * padding)

    return np.array(result, dtype=int)