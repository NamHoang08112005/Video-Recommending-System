"""Utilities to load processed datasets and visualize collaborative features."""

import pickle
from pathlib import Path

import pandas as pd
from scipy import sparse

try:
    from .CollaborativeFeatureEngineer import CollaborativeFeatureEngineer
except ImportError:
    from CollaborativeFeatureEngineer import CollaborativeFeatureEngineer


def load_processed_data(processed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed movies and ratings parquet files from data/processed."""
    if processed_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"

    movies_path = processed_dir / "clean_movies.parquet"
    ratings_path = processed_dir / "clean_ratings.parquet"

    if not movies_path.exists() or not ratings_path.exists():
        missing = []
        if not movies_path.exists():
            missing.append(str(movies_path))
        if not ratings_path.exists():
            missing.append(str(ratings_path))
        raise FileNotFoundError("Khong tim thay file du lieu da xu ly:\n- " + "\n- ".join(missing))

    movies_df = pd.read_parquet(movies_path)
    ratings_df = pd.read_parquet(ratings_path)
    return movies_df, ratings_df


def show_processed_data(processed_dir: Path | None = None, n_rows: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Display shape and top rows from processed datasets."""
    movies_df, ratings_df = load_processed_data(processed_dir)

    print("=== PROCESSED DATA PREVIEW ===")
    print(f"Movies shape : {movies_df.shape}")
    print(f"Ratings shape: {ratings_df.shape}")

    movie_preview_cols = ["id", "title", "genres_text", "content_feature"]
    available_cols = [col for col in movie_preview_cols if col in movies_df.columns]

    print(f"\nTop {n_rows} rows - Movies:")
    print(movies_df[available_cols].head(n_rows).to_string(index=False))

    print(f"\nTop {n_rows} rows - Ratings:")
    print(ratings_df.head(n_rows).to_string(index=False))
    return movies_df, ratings_df


def show_collaborative_visualizations(processed_dir: Path | None = None, heatmap_k: int = 50) -> None:
    """Fit collaborative features and show both distribution and top-interaction heatmap."""
    _, ratings_df = load_processed_data(processed_dir)

    engineer = CollaborativeFeatureEngineer()
    engineer.fit_transform(ratings_df)

    # Ensure k is valid for current matrix dimensions.
    max_k = min(engineer.matrix_normalized.shape)
    k = max(1, min(heatmap_k, max_k))

    engineer.visualize_distribution(ratings_df["rating"])
    engineer.plot_top_interactions_heatmap(k=k)


def show_collaborative_artifacts(processed_dir: Path | None = None, n_samples: int = 10) -> None:
    """Print summary of collaborative artifacts saved in data/processed."""
    if processed_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"

    mapping_path = processed_dir / "collab_mappings.pkl"
    matrix_path = processed_dir / "collab_matrix_normalized.npz"

    if not mapping_path.exists() or not matrix_path.exists():
        missing = []
        if not mapping_path.exists():
            missing.append(str(mapping_path))
        if not matrix_path.exists():
            missing.append(str(matrix_path))
        raise FileNotFoundError("Khong tim thay collaborative artifacts:\n- " + "\n- ".join(missing))

    with open(mapping_path, "rb") as f:
        meta = pickle.load(f)
    matrix = sparse.load_npz(matrix_path)

    user_mapping = meta.get("user_mapping", {})
    movie_mapping = meta.get("movie_mapping", {})

    print("\n=== COLLABORATIVE ARTIFACTS PREVIEW ===")
    print(f"Matrix path : {matrix_path}")
    print(f"Mapping path: {mapping_path}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix nnz  : {matrix.nnz}")
    print(f"Matrix dtype: {matrix.dtype}")

    sparsity = (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
    print(f"Sparsity    : {sparsity:.4f}%")

    print(f"User mappings : {len(user_mapping)}")
    print(f"Movie mappings: {len(movie_mapping)}")
    print(f"Sample users  : {list(user_mapping.items())[:n_samples]}")
    print(f"Sample movies : {list(movie_mapping.items())[:n_samples]}")

    user_means = meta.get("user_means")
    item_means = meta.get("item_means")
    print(f"user_means shape: {getattr(user_means, 'shape', None)}")
    print(f"item_means shape: {getattr(item_means, 'shape', None)}")


if __name__ == "__main__":
    # show_processed_data(n_rows=5)
    show_collaborative_artifacts(n_samples=10)
    # show_collaborative_visualizations(heatmap_k=50)
