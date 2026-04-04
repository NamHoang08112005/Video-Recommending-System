"""Content-based recommendation logic using cosine similarity."""


def top_k_similar(similarity_row, k=10):
    """Return indices of top-k similar items excluding itself."""
    import numpy as np

    ranked_idx = np.argsort(similarity_row)[::-1]
    return [idx for idx in ranked_idx if similarity_row[idx] < 0.999999][:k]
