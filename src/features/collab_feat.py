"""Collaborative feature engineering (user-item sparse matrices)."""


def build_user_item_matrix(df, user_col="user_id", item_col="item_id", rating_col="rating"):
    """Build a COO sparse matrix from interaction logs."""
    from scipy.sparse import coo_matrix

    users = df[user_col].astype("category")
    items = df[item_col].astype("category")
    ratings = df[rating_col].astype(float)

    return coo_matrix(
        (ratings, (users.cat.codes, items.cat.codes)),
        shape=(users.cat.categories.size, items.cat.categories.size),
    )
