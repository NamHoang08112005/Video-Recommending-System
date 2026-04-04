"""Data loading utilities for the recommendation system."""

from pathlib import Path


def load_csv(path: str):
    """Load a CSV file from a relative or absolute path."""
    import pandas as pd

    csv_path = Path(path)
    return pd.read_csv(csv_path)
