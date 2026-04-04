"""Hybrid recommendation engine combining content and collaborative scores."""


def blend_scores(content_scores, collab_scores, alpha=0.5):
    """Blend two score vectors with weight alpha."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    content_norm = scaler.fit_transform(content_scores.reshape(-1, 1)).ravel()
    collab_norm = scaler.fit_transform(collab_scores.reshape(-1, 1)).ravel()
    return alpha * content_norm + (1 - alpha) * collab_norm
