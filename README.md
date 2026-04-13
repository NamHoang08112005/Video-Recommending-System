# Video Recommendation System

Project scaffold for a hybrid recommender system (content-based + collaborative filtering).

## Overall Pipeline

![Overall pipeline](docs/images/Pipeline.png)

## Tasks cần làm 

### Ngày 1: Data Selection & Exploratory Data Analysis (EDA)

Mục tiêu kỹ thuật:
- Làm sạch dữ liệu, hiểu phân phối tương tác và xác định các feature cốt lõi cần trích xuất.

Kỹ thuật/Thuật toán:
- Xử lý Missing Values: Drop các bản ghi thiếu metadata quan trọng (`title`, `overview`).
- Xử lý Outliers: Lọc bỏ user có quá ít tương tác (dưới 5 đánh giá) để giảm nhiễu ma trận.
- Thống kê mô tả trên distribution của rating/view.

Ước lượng thời gian:
- 1 ngày.

Tasks:
- [ ] Load và hợp nhất dữ liệu thô từ `data/raw/`.
- [ ] Kiểm tra tỷ lệ missing cho các cột quan trọng.
- [ ] Drop bản ghi thiếu `title` hoặc `overview`.
- [ ] Tính số lượng tương tác theo user, lọc user có `< 5` interactions.
- [ ] Vẽ/ghi nhận phân phối rating và số interactions mỗi user.
- [ ] Lưu dữ liệu đã làm sạch vào `data/processed/`.

### Ngày 2: Feature Engineering (Content-based)

Mục tiêu kỹ thuật:
- Chuyển đổi dữ liệu văn bản thành vector đặc trưng trong không gian toán học.

Kỹ thuật/Thuật toán:
- Text preprocessing: lowercasing, stopword removal (NLTK), stemming/lemmatization.
- Ghép `title`, `overview`, `genres` thành một chuỗi `soup`.
- Dùng TF-IDF Vectorizer với `min_df`, `max_df` phù hợp.

Ước lượng thời gian:
- 1 ngày.

Tasks:
- [ ] Xây dựng hàm tiền xử lý văn bản trong `src/features/content_feat.py`.
- [ ] Tạo cột `soup` từ metadata.
- [ ] Fit TF-IDF trên tập train/processed.
- [ ] Tune `min_df`, `max_df`, `ngram_range` theo kích thước dữ liệu.
- [ ] Lưu vectorizer vào `artifacts/tfidf_vectorizer.pkl`.

### Ngày 3: Feature Engineering (Collaborative)

Mục tiêu kỹ thuật:
- Xây dựng ma trận tương tác User-Item tối ưu cho tính toán.

Kỹ thuật/Thuật toán:
- Pivot logs thành User-Item Interaction Matrix.
- Mean-centering để giảm user bias.
- Dùng định dạng sparse matrix (CSR) của `scipy`.

Ước lượng thời gian:
- 1 ngày.

Tasks:
- [ ] Tạo user-item matrix từ logs trong `src/features/collab_feat.py`.
- [ ] Cài mean-centering theo user.
- [ ] Chuyển ma trận sang CSR để tối ưu RAM.
- [ ] Lưu matrix/metadata mapping sang `data/processed/`.

### Ngày 4: Model Training (Content-based Filtering)

Mục tiêu kỹ thuật:
- Tạo bảng tra cứu độ tương đồng giữa các video dựa trên nội dung.

Kỹ thuật/Thuật toán:
- Tính Cosine Similarity hoặc Linear Kernel từ ma trận TF-IDF.
- Pre-compute top 50 video tương đồng nhất cho mỗi video để truy vấn nhanh.

Ước lượng thời gian:
- 1 ngày.

Tasks:
- [ ] Tính similarity matrix từ TF-IDF.
- [ ] Sinh top-50 neighbors cho mỗi video.
- [ ] Đóng gói logic truy vấn trong `src/models/content_based.py`.
- [ ] Lưu artifact vào `artifacts/similarity_matrix.npz` hoặc dictionary serialized.

### Ngày 5: Model Training (Collaborative Filtering)

Mục tiêu kỹ thuật:
- Dự đoán điểm tương tác của user cho các video chưa xem.

Kỹ thuật/Thuật toán:
- Dùng Truncated SVD (Matrix Factorization) để học latent features.
- Tùy chọn thay thế: Item-based KNN (Pearson/Cosine) nếu ma trận quá thưa.
- Khuyến nghị dùng thư viện `surprise` để rút ngắn thời gian implement.

Ước lượng thời gian:
- 1 ngày.

Tasks:
- [ ] Huấn luyện mô hình SVD trong `src/models/collaborative.py`.
- [ ] Validate nhanh bằng RMSE trên validation split.
- [ ] Nếu kết quả kém do sparsity, thử Item-KNN baseline.
- [ ] Lưu model vào `artifacts/svd_model.pkl`.

### Ngày 6: Tích hợp Hybrid System

Mục tiêu kỹ thuật:
- Ghép 2 mô hình đã train thành pipeline duy nhất để sinh recommend cuối cùng.

Kỹ thuật/Thuật toán:
- Chạy song song 2 nhánh để lấy Top-N candidates.
- Chuẩn hóa điểm về [0, 1] bằng Min-Max Scaler.
- Kết hợp bằng trọng số $\alpha$ với baseline $\alpha = 0.5$.
- Xây dựng hàm `recommend(user_id, video_id, alpha)`.

Ước lượng thời gian:
- 1 ngày.

Tasks:
- [ ] Hoàn thiện hàm blend score trong `src/pipeline/hybrid_engine.py`.
- [ ] Triển khai `recommend(user_id, video_id, alpha)`.
- [ ] Kiểm tra nhiều giá trị alpha: 0.3 / 0.5 / 0.7.
- [ ] Đảm bảo trả về Top-K đúng format.

### Ngày 7: Evaluation & Refinement

Mục tiêu kỹ thuật:
- Đo lường độ chính xác và khả năng xếp hạng offline.

Kỹ thuật/Thuật toán:
- Tách Train/Test theo thời gian (time-based split) hoặc leave-one-out.
- Ranking metrics: NDCG@K, MAP@K.
- Rating metric: RMSE cho nhánh Collaborative.

Ước lượng thời gian:
- 1 ngày.

Tasks:
- [ ] Tạo pipeline đánh giá trong `src/evaluation/metrics.py`.
- [ ] Tính NDCG@K và MAP@K cho kết quả recommend.
- [ ] Tính RMSE cho dự đoán rating.
- [ ] So sánh Content-only vs Collab-only vs Hybrid.
- [ ] Tổng hợp báo cáo kết quả và đề xuất tinh chỉnh tiếp theo.





## Structure

- `configs/`: YAML configurations for data and model settings.
- `data/raw/`: immutable source data (raw CSV files).
- `data/processed/`: cleaned data and feature matrices.
- `artifacts/`: precomputed similarity matrices and training artifacts.
- `notebooks/`: exploratory data analysis and model training notebooks.

```text
video_rec_system/
├── configs/
│   ├── data_config.yaml
│   └── model_config.yaml
├── data/
│   ├── raw/
│   │   ├── credits.csv
│   │   ├── keywords.csv
│   │   ├── links_small.csv
│   │   ├── links.csv
│   │   ├── movies_metadata.csv
│   │   ├── ratings_small.csv
│   │   └── ratings.csv
│   └── processed/
│       ├── cold_movies_matrix.npz
│       ├── cold_ratings_matrix.npz
│       ├── movies_matrix.npz
│       ├── ratings_matrix.npz
│       ├── test_warm_ratings_matrix.npz
│       ├── train_warm_movies_matrix.npz
│       └── train_warm_ratings_matrix.npz
├── artifacts/
│   ├── similarity_matrix.npz
│   ├── logs/
│   └── split_cache/
│       ├── tracked_test.npz
│       ├── train.npz
│       └── untracked_test.npz
├── docs/
│   └── images/
├── notebooks/
│   ├── 01_Cleaning_Data.ipynb
│   ├── 02_Finding similar videos.ipynb
│   └── 03_Trainning user_based_recommeding_model.ipynb
├── main.py
├── requirements.txt
└── README.md
```

## Quick start

1. Create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment:

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bash
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python main.py
```


## Thông tin dữ liệu sau khi xử lý 