import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    if not documents:
        return np.array([]), []

    N = len(documents)
    
    doc_tokens = [doc.lower().split() for doc in documents]

    unique_words = set()
    for tokens in doc_tokens:
        unique_words.update(tokens)

    vocab = sorted(list(unique_words))
    vocab_size = len(vocab)

    if vocab_size == 0:
        return np.array([]), []

    word_to_idx = {word: i for i, word in enumerate(vocab)}

    doc_freqs = Counter()
    for tokens in doc_tokens:
        unique_tokens_in_doc = set(tokens)
        for token in unique_tokens_in_doc:
            doc_freqs[token] += 1
    
    # note doc_freqs[token] will never be 0 because vocab is built from documents
    idf_vector = np.zeros(vocab_size)
    for i, word in enumerate(vocab):
        df = doc_freqs[word]
        idf_vector[i] = math.log(N / df)
    
    tfidf_matrix = np.zeros((N, vocab_size))

    for doc_idx, tokens in enumerate(doc_tokens):
        total_terms = len(tokens)
        if total_terms == 0:
            continue
        term_counts = Counter(tokens)

        for word, count in term_counts.items():
            if word in word_to_idx:
                word_idx = word_to_idx[word]

                tf = count / total_terms

                tfidf_matrix[doc_idx, word_idx] = tf * idf_vector[word_idx]
    
    return (tfidf_matrix, vocab)


