import numpy as np
from collections import Counter 
def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    ctr = Counter(tokens)
    ans = np.zeros((len(vocab), ), dtype=int)
    for i, word in enumerate(vocab):
        ans[i] = ctr[word]
    
    return ans 