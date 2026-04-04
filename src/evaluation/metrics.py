"""Evaluation metrics for recommender systems."""


def rmse(y_true, y_pred):
    """Compute RMSE."""
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ndcg_at_k(relevance, k=10):
    """Compute NDCG@K from a relevance list ordered by predicted rank."""
    import numpy as np

    rel = np.asfarray(relevance)[:k]
    if rel.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, rel.size + 2))
    dcg = np.sum((2**rel - 1) / discounts)

    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2**ideal - 1) / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0
