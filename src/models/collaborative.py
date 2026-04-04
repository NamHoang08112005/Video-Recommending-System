"""Collaborative filtering model training and inference."""


def train_svd(trainset, n_factors=100, n_epochs=20, random_state=42):
    """Train a Surprise SVD model."""
    from surprise import SVD

    model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
    model.fit(trainset)
    return model
