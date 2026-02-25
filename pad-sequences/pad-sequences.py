import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.array([])
    
    if max_len is None:
        max_len=max(len(s) for s in seqs)
    arr=np.full((len(seqs),max_len),pad_value)
    #starts with 0
    for i in range(len(seqs)):
        seq=seqs[i]
        for j in range(min(len(seq),max_len)):
            arr[i][j]=seq[j] #fills the elements in the zero array 
    return arr
    pass