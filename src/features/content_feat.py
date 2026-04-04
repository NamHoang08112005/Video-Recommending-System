"""Content feature engineering (TF-IDF, text preprocessing)."""


def build_tfidf_features(text_series):
    """Fit TF-IDF on text input and return sparse matrix + vectorizer."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(text_series)
    return matrix, vectorizer
