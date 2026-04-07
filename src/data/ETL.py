import pandas as pd
import numpy as np
import ast
import gc # Garbage Collector để dọn rác RAM chủ động
import pickle
from pathlib import Path
from scipy import sparse

try:
    from .CollaborativeFeatureEngineer import CollaborativeFeatureEngineer
except ImportError:
    from CollaborativeFeatureEngineer import CollaborativeFeatureEngineer

# Mô tả dữ liệu được xử lý:
# movies_df: DataFrame chứa thông tin phim đã được làm sạch, với các cột như id, title, genres_text, keywords, credits, content_feature (chuỗi tổng hợp cho Content-Based)
# ratings_df: DataFrame chứa thông tin đánh giá đã được cắt tỉa, với các cột userId, movieId, rating (đã ép kiểu int32/float32 để tiết kiệm RAM)
# id là khóa chính để liên kết giữa movies_df và ratings_df. Các cột genres_text, keywords, credits đã được trích xuất và làm sạch để phục vụ cho mô hình Content-Based Filtering. Ma trận User-Item đã được cắt tỉa để loại bỏ các user vãng lai và phim quá chìm, giúp tăng hiệu quả của mô hình Collaborative Filtering.
# genres_text: Chuỗi chứa tên các thể loại phim, được trích xuất từ cột genres gốc (dạng string của list of dicts).
# keywords: Chuỗi chứa tên các từ khóa liên quan đến phim (Nội dung phim)
# credits: Chuỗi chứa tên các diễn viên chính và đạo diễn (Thông tin về dàn diễn viên và đạo diễn)
# content_feature: Chuỗi tổng hợp từ genres_text + keywords + credits, được sử dụng làm đặc trưng đầu vào cho mô hình Content-Based Filtering.
# userId: ID của người dùng (đã ép kiểu int32)
# movieId: ID của phim (đã ép kiểu int32)
# rating: Điểm đánh giá của user cho movie (đã ép kiểu float32)


class MovieDataPipeline:
    def __init__(self, metadata_path, ratings_path, keywords_path=None, credits_path=None):
        """
        Khởi tạo Pipeline với đường dẫn file.
        """
        self.metadata_path = metadata_path
        self.ratings_path = ratings_path
        self.keywords_path = keywords_path
        self.credits_path = credits_path
        self.movies_df = None
        self.ratings_df = None
        self.collab_matrix = None

    @staticmethod
    def _safe_literal_eval(value):
        if pd.isna(value):
            return []
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    @staticmethod
    def _extract_names(json_like_str):
        items = MovieDataPipeline._safe_literal_eval(json_like_str)
        return ' '.join(str(item.get('name', '')).strip() for item in items if item.get('name'))

    @staticmethod
    def _extract_credits_text(cast_str, crew_str):
        cast_items = MovieDataPipeline._safe_literal_eval(cast_str)
        crew_items = MovieDataPipeline._safe_literal_eval(crew_str)

        top_cast = [
            str(member.get('name', '')).strip()
            for member in cast_items[:5]
            if member.get('name')
        ]
        directors = [
            str(member.get('name', '')).strip()
            for member in crew_items
            if str(member.get('job', '')).lower() == 'director' and member.get('name')
        ]
        return ' '.join(top_cast + directors)

    def load_and_optimize(self):
        """
        Nạp dữ liệu kèm kỹ thuật Read-time Downcasting (Ép kiểu ngay khi đọc).
        Đây là kỹ thuật sống còn để chạy dữ liệu lớn trên laptop.
        """
        print("1. Đang nạp và tối ưu hóa bộ nhớ...")
        
        # Chỉ nạp những cột thật sự cần thiết. Bỏ qua các cột rác như homepage, tagline...
        metadata_cols = ['id', 'title', 'genres']
        
        # Đọc metadata, xử lý lỗi dòng (error_bad_lines) do file csv gốc thỉnh thoảng lỗi format
        self.movies_df = pd.read_csv(self.metadata_path, usecols=metadata_cols, low_memory=False)
        
        # Xử lý cột ID của movies: Loại bỏ các ID không phải là số (dữ liệu nhiễu) và ép về int32
        self.movies_df['id'] = pd.to_numeric(self.movies_df['id'], errors='coerce')
        self.movies_df = self.movies_df.dropna(subset=['id'])
        self.movies_df['id'] = self.movies_df['id'].astype('int32')

        # Nạp và tạo cột keywords từ keywords.csv
        if self.keywords_path is not None and Path(self.keywords_path).exists():
            keywords_df = pd.read_csv(self.keywords_path, usecols=['id', 'keywords'], low_memory=False)
            keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')
            keywords_df = keywords_df.dropna(subset=['id'])
            keywords_df['id'] = keywords_df['id'].astype('int32')
            keywords_df['keywords'] = keywords_df['keywords'].apply(self._extract_names)
            self.movies_df = self.movies_df.merge(keywords_df[['id', 'keywords']], on='id', how='left')
        else:
            self.movies_df['keywords'] = ''

        # Nạp và tạo cột credits từ credits.csv (gồm top cast + director)
        if self.credits_path is not None and Path(self.credits_path).exists():
            credits_df = pd.read_csv(self.credits_path, usecols=['id', 'cast', 'crew'], low_memory=False)
            credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
            credits_df = credits_df.dropna(subset=['id'])
            credits_df['id'] = credits_df['id'].astype('int32')
            credits_df['credits'] = credits_df.apply(
                lambda row: self._extract_credits_text(row['cast'], row['crew']), axis=1
            )
            self.movies_df = self.movies_df.merge(credits_df[['id', 'credits']], on='id', how='left')
        else:
            self.movies_df['credits'] = ''

        self.movies_df['keywords'] = self.movies_df['keywords'].fillna('')
        self.movies_df['credits'] = self.movies_df['credits'].fillna('')

        # Đọc Ratings: Chỉ định trước Data Types (Kỹ thuật ăn tiền với dân C++)
        # Thay vì tốn 64 bits cho rating, ta dùng float32. user_id và movie_id dùng int32
        rating_dtypes = {
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float32'
        }
        # Chỉ lấy 3 cột, bỏ qua 'timestamp' vì RecSys cơ bản chưa dùng tới Context-aware
        self.ratings_df = pd.read_csv(self.ratings_path, usecols=['userId', 'movieId', 'rating'], dtype=rating_dtypes)
        
        print(f"-> Đã nạp thành công! Ratings shape: {self.ratings_df.shape}")

    def clean_metadata(self):
        """
        Xử lý Missing Values và làm sạch Text cho Content-Based.
        """
        print("2. Đang làm sạch Metadata...")
        
        # Đảm bảo title luôn hợp lệ cho downstream hiển thị/recommendation.
        self.movies_df = self.movies_df.dropna(subset=['title'])
        
        # Cột genres trong dataset này đang ở dạng String của List of Dictionaries (VD: "[{'id': 12, 'name': 'Animation'}]")
        # Ta cần parse string này thành Python object bằng ast.literal_eval, sau đó rút trích tên thể loại.
        def extract_genres(x):
            try:
                # Trích xuất 'name' từ list các dict
                genres = ast.literal_eval(x)
                return ' '.join([i['name'] for i in genres])
            except:
                return ''
                
        self.movies_df['genres_text'] = self.movies_df['genres'].apply(extract_genres)
        self.movies_df = self.movies_df.drop(columns=['genres']) # Xóa cột cũ giải phóng RAM

    def prune_interactions(self, min_movie_ratings=10, min_user_ratings=5):
        """
        Cắt tỉa ma trận User-Item. Loại bỏ Users vãng lai và Phim quá chìm.
        """
        print(f"3. Đang cắt tỉa ma trận (Lọc User >= {min_user_ratings} rates, Movie >= {min_movie_ratings} rates)...")
        
        # Bước 1: Giữ các phim có từ min_movie_ratings đánh giá trở lên
        movie_counts = self.ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(valid_movies)]
        
        # Bước 2: Giữ các user có từ min_user_ratings đánh giá trở lên
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
        
        # Chỉ định lại index để dọn dẹp RAM dư thừa từ Pandas View
        self.ratings_df = self.ratings_df.reset_index(drop=True)
        gc.collect() # Ép Garbage Collector chạy tay để dọn dẹp biến tạm
        
        print(f"-> Sau cắt tỉa: Ratings shape thu gọn còn {self.ratings_df.shape}")

    def extract_features(self):
        """
        Chuẩn bị Feature cuối cùng cho 2 nhánh mô hình.
        """
        print("4. Chuẩn bị đặc trưng (Feature Extraction)...")
        
        # Đảm bảo dữ liệu Ratings chỉ tham chiếu tới các Phim có tồn tại trong Metadata
        valid_metadata_ids = self.movies_df['id'].unique()
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(valid_metadata_ids)]
        
        # Content-Based Feature: Gộp Text
        # Content-based models (như TF-IDF hoặc Word2Vec) cần 1 chuỗi văn bản tổng hợp.
        self.movies_df['content_feature'] = (
            self.movies_df['genres_text'] + " " + self.movies_df['keywords'] + " " + self.movies_df['credits']
        )
        
    def show_statistics(self):
        """
        In ra Thống kê mô tả (Descriptive Statistics).
        """
        print("\n=== THỐNG KÊ MÔ TẢ TỔNG QUAN ===")
        print("1. Phân phối điểm đánh giá (Ratings Distribution):")
        print(self.ratings_df['rating'].describe())
        
        sparsity = 1.0 - (len(self.ratings_df) / (self.ratings_df['userId'].nunique() * self.ratings_df['movieId'].nunique()))
        print(f"\n2. Độ thưa của ma trận (Sparsity): {sparsity * 100:.4f}%")
        print("   (Điều này có nghĩa là {:.4f}% ô trong ma trận đang bị trống - Rất bình thường trong RecSys!)".format(sparsity * 100))

    def build_collaborative_artifacts(self, output_dir):
        """
        Dùng CollaborativeFeatureEngineer để tạo ma trận thưa chuẩn hóa
        và lưu artifacts phục vụ huấn luyện/inference collaborative.
        """
        print("5. Đang tối ưu dữ liệu Collaborative bằng CollaborativeFeatureEngineer...")

        engineer = CollaborativeFeatureEngineer()
        self.collab_matrix = engineer.fit_transform(self.ratings_df)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        collab_matrix_path = output_dir / 'collab_matrix_normalized.npz'
        collab_meta_path = output_dir / 'collab_mappings.pkl'

        sparse.save_npz(collab_matrix_path, self.collab_matrix)
        with open(collab_meta_path, 'wb') as f:
            pickle.dump(engineer.get_meta_data(), f)

        print(f"-> Collaborative matrix shape: {self.collab_matrix.shape}, nnz={self.collab_matrix.nnz}")
        print(f"-> Sparsity sau chuẩn hóa: {engineer.get_sparsity():.4f}%")
        print(f"-> Đã lưu collaborative artifacts:\n- Matrix: {collab_matrix_path}\n- Metadata: {collab_meta_path}")

# --- CÁCH SỬ DỤNG PIPELINE ---
if __name__ == "__main__":
    # Tự động suy ra root project để chạy được từ mọi thư mục hiện hành
    project_root = Path(__file__).resolve().parents[2]
    METADATA_PATH = project_root / 'data' / 'raw' / 'movies_metadata.csv'
    RATINGS_PATH = project_root / 'data' / 'raw' / 'ratings.csv'  # Có thể thử với ratings_small.csv trước để test logic
    KEYWORDS_PATH = project_root / 'data' / 'raw' / 'keywords.csv'
    CREDITS_PATH = project_root / 'data' / 'raw' / 'credits.csv'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo và chạy tuần tự
    pipeline = MovieDataPipeline(METADATA_PATH, RATINGS_PATH, KEYWORDS_PATH, CREDITS_PATH)
    pipeline.load_and_optimize()
    pipeline.clean_metadata()
    pipeline.prune_interactions(min_movie_ratings=10, min_user_ratings=5)
    pipeline.extract_features()
    pipeline.show_statistics()
    pipeline.build_collaborative_artifacts(processed_dir)
    
    # Mẹo nhỏ: Lưu lại dưới dạng parquet (nhanh hơn, lưu trữ cấu trúc type int32/float32 tốt hơn CSV)
    clean_ratings_path = processed_dir / 'clean_ratings.parquet'
    clean_movies_path = processed_dir / 'clean_movies.parquet'
    pipeline.ratings_df.to_parquet(clean_ratings_path)
    pipeline.movies_df.to_parquet(clean_movies_path)
    print(f"\n6. Đã lưu dữ liệu xử lý:")
    print(f"- Ratings: {clean_ratings_path}")
    print(f"- Movies: {clean_movies_path}")