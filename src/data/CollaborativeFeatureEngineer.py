import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional

class CollaborativeFeatureEngineer:
    """
    Module xử lý Feature Engineering cho hệ thống gợi ý sử dụng ma trận thưa.
    Tối ưu hóa cho dữ liệu lớn và bộ nhớ RAM hạn chế.
    """

    def __init__(self):
        self.user_to_idx: Dict = {}
        self.idx_to_user: Dict = {}
        self.movie_to_idx: Dict = {}
        self.idx_to_movie: Dict = {}
        self.user_means: Optional[np.ndarray] = None
        self.item_means: Optional[np.ndarray] = None
        self.matrix_normalized: Optional[csr_matrix] = None

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Thực hiện toàn bộ quy trình từ mapping đến chuẩn hóa.
        
        Args:
            df: DataFrame chứa columns ['userId', 'movieId', 'rating']
            
        Returns:
            csr_matrix: Ma trận đã được chuẩn hóa (Double-centered).
        """
        # Bước 1: Mapping IDs
        row_indices, col_indices = self._create_mappings(df)
        
        # Bước 2: Build Sparse Matrix
        # Sử dụng float32 để tiết kiệm 50% RAM so với float64
        ratings = df['rating'].values.astype(np.float32)
        sparse_mat = coo_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.movie_to_idx))
        ).tocsr()

        # Bước 3: Double-Centering
        self.matrix_normalized = self._apply_double_centering(sparse_mat)
        
        return self.matrix_normalized

    def _create_mappings(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo từ điển ánh xạ hai chiều và trả về mảng index."""
        unique_users = df['userId'].unique()
        unique_movies = df['movieId'].unique()

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        
        self.movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie = {i: mid for mid, i in self.movie_to_idx.items()}

        row_indices = df['userId'].map(self.user_to_idx).values.astype(np.int32)
        col_indices = df['movieId'].map(self.movie_to_idx).values.astype(np.int32)
        
        return row_indices, col_indices

    def _apply_double_centering(self, matrix: csr_matrix) -> csr_matrix:
        """Khử bias User và Item trên cấu trúc thưa."""
        # Chế độ tính toán tránh biến đổi ma trận thưa thành ma trận dày (dense)
        mat = matrix.copy()
        
        # 1. Row-wise mean centering (User bias)
        # Chỉ tính trung bình trên các phần tử khác 0
        row_sums = mat.sum(axis=1).A1
        row_counts = np.diff(mat.indptr)
        self.user_means = np.divide(row_sums, row_counts,out=np.zeros_like(row_sums, dtype=np.float32), where=row_counts > 0)
        
        # Trừ trung bình hàng
        mat.data -= np.repeat(self.user_means, row_counts)

        # 2. Column-wise mean centering (Item bias)
        mat_csc = mat.tocsc()
        col_sums = mat_csc.sum(axis=0).A1
        col_counts = np.diff(mat_csc.indptr)
        self.item_means = np.divide(col_sums, col_counts,out=np.zeros_like(col_sums, dtype=np.float32), where=col_counts > 0)
        
        # Trừ trung bình cột
        mat_csc.data -= np.repeat(self.item_means, col_counts)
        
        return mat_csc.tocsr()

    def get_sparsity(self) -> float:
        """Tính toán phần trăm độ thưa của ma trận."""
        if self.matrix_normalized is None:
            return 0.0
        n_elements = self.matrix_normalized.shape[0] * self.matrix_normalized.shape[1]
        n_nonzero = self.matrix_normalized.nnz
        sparsity = (1 - n_nonzero / n_elements) * 100
        return sparsity

    def visualize_distribution(self, original_ratings: pd.Series):
        """Vẽ biểu đồ phân phối trước và sau chuẩn hóa."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(original_ratings, bins=10, kde=True, color='blue')
        plt.title("Original Rating Distribution")
        
        plt.subplot(1, 2, 2)
        sns.histplot(self.matrix_normalized.data, bins=50, kde=True, color='red')
        plt.title("Double-Centered Rating Distribution")
        plt.show()

    def plot_top_interactions_heatmap(self, k: int = 50):
        """Vẽ Heatmap cho Top-K User và Item có nhiều tương tác nhất."""
        # Tìm top users/items dựa trên số lượng tương tác (nnz)
        row_counts = np.diff(self.matrix_normalized.indptr)
        top_users = np.argsort(row_counts)[-k:]
        
        col_counts = np.diff(self.matrix_normalized.tocsc().indptr)
        top_items = np.argsort(col_counts)[-k:]
        
        # Trích xuất ma trận con (Sub-matrix)
        sub_mat = self.matrix_normalized[top_users, :][:, top_items].toarray()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sub_mat, cmap='RdBu_r', center=0)
        plt.title(f"Heatmap of Top {k}x{k} Interactions (Normalized)")
        plt.xlabel("Movie Index")
        plt.ylabel("User Index")
        plt.show()

    def get_meta_data(self) -> Dict:
        """Xuất dữ liệu mapping và các giá trị bias."""
        return {
            "user_mapping": self.user_to_idx,
            "movie_mapping": self.movie_to_idx,
            "user_means": self.user_means,
            "item_means": self.item_means
        }