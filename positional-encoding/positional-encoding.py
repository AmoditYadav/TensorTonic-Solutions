import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    # positions: (seq_len, 1)
    positions = np.arange(seq_len, dtype=float)[:, None]

    # number of frequency pairs (ceil so odd d_model works)
    half = (d_model + 1) // 2

    # frequency indices
    freqs = np.arange(half, dtype=float)[None, :]

    # compute angles
    angles = positions / (base ** (2 * freqs / d_model))

    # allocate output
    pe = np.zeros((seq_len, d_model), dtype=float)

    # fill sin (even columns)
    pe[:, 0::2] = np.sin(angles[:, :pe[:, 0::2].shape[1]])

    # fill cos (odd columns)
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])

    return pe
    