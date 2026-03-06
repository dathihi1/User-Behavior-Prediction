# DATAFLOW 2026 — Dự Đoán Hành Vi Người Dùng

Mô hình phân loại đa đầu ra dự đoán đồng thời 6 thuộc tính hành vi độc lập từ chuỗi hành động lịch sử của người dùng.

---

## Tổng Quan Bài Toán

| Thông tin | Chi tiết |
|-----------|---------|
| Đầu vào | Chuỗi hành động người dùng có độ dài thay đổi (tối đa 37 bước) |
| Đầu ra | 6 thuộc tính: `attr_1`–`attr_6` (uint16) |
| Metric chính | **Exact-Match Accuracy** — cả 6 thuộc tính phải đúng đồng thời |
| Dữ liệu huấn luyện | 51.000 chuỗi × 37 cột hành động |
| Dữ liệu validation | 7.200 chuỗi (chỉ dùng đánh giá offline) |
| Dữ liệu test | 38.000 chuỗi (không có nhãn) |
| Số hành động duy nhất | 254 (ID thưa từ 102–24.438, ánh xạ dày về 1–254) |
| Số lớp mục tiêu | attr_1=12, attr_2=31, attr_3=99, attr_4=12, attr_5=31, attr_6=99 |

---

## Quy Tắc Tách Dữ Liệu — BẮT BUỘC TUYỆT ĐỐI

**Train và Validation KHÔNG BAO GIỜ được gộp lại — không có ngoại lệ.**

- Nhãn `Y_val` KHÔNG được dùng trong bất kỳ bước huấn luyện nào.
- `Y_val` KHÔNG được dùng cho early stopping — chỉ dùng fold nội bộ 90/10 từ `X_train`.
- `merge_train_val()` KHÔNG được gọi. Không có cờ `--final`.
- Submission được tạo bằng cách predict `X_test` với mô hình huấn luyện trên `X_train` mà thôi.

---

## Cấu Trúc Dự Án

```
user-behavior-prediction/
├── configs/
│   └── config.yaml                  # Toàn bộ siêu tham số (nguồn duy nhất)
├── notebooks/
│   ├── 01_eda.ipynb                 # Phân tích dữ liệu EDA (chỉ dùng X_train / Y_train)
│   ├── 02_feature_engineering.ipynb # Khám phá đặc trưng
│   └── 03_modeling.ipynb            # Huấn luyện, so sánh mô hình, sinh submission
├── outputs/
│   ├── figures/                     # Biểu đồ EDA và so sánh mô hình
│   ├── models/                      # Mô hình đã lưu (.pkl)
│   └── submissions/                 # File CSV nộp bài
├── src/
│   ├── data/
│   │   ├── loader.py                # DataLoader (CSV → DataFrame)
│   │   └── preprocessor.py          # SequencePreprocessor (remap + pad), TargetEncoder
│   ├── features/
│   │   ├── sequence_features.py     # TF-IDF, N-gram
│   │   ├── statistical_features.py  # Statistical + HistogramFeatureExtractor
│   │   └── feature_pipeline.py      # Kết hợp toàn bộ đặc trưng ML (~4.000 chiều)
│   ├── models/
│   │   ├── base_model.py            # Giao diện BaseModel trừu tượng
│   │   ├── xgboost_model.py         # XGBoostMultiOutput (GPU hist)
│   │   ├── lstm_model.py            # LSTMMultiOutput (BiLSTM + Attention, AMP)
│   │   ├── transformer_model.py     # TransformerMultiOutput (CLS token, AMP)
│   │   └── ensemble_model.py        # EnsembleMultiOutput (weighted soft-voting)
│   ├── evaluation/
│   │   └── metrics.py               # exact_match_accuracy, per_attribute_f1, analyze_errors
│   └── utils/
│       ├── seed.py                  # set_seed (numpy + torch + random)
│       └── helpers.py               # load_config, get_logger, save/load_pickle
├── requirements.txt
└── README.md
```

> **Lưu ý:** Toàn bộ quy trình huấn luyện, đánh giá và sinh submission được thực hiện trong
> `notebooks/03_modeling.ipynb`. Không có file script CLI riêng lẻ.

---

## Cài Đặt Môi Trường

### 1. Tạo môi trường ảo

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac
```

### 2. Cài PyTorch với CUDA (thực hiện trước)

```bash
# CUDA 12.1 — tương thích với PyTorch 2.5.1
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Dự phòng CPU (chậm hơn nhiều với LSTM / Transformer):
# pip install torch>=2.0.0
```

### 3. Cài toàn bộ thư viện còn lại

```bash
pip install -r requirements.txt
```

---

## Chuẩn Bị Dữ Liệu

Đặt các file CSV của cuộc thi vào thư mục `../data/` (cùng cấp với thư mục dự án):

```
data/
├── X_train.csv   Y_train.csv   # 51.000 dòng
├── X_val.csv     Y_val.csv     # 7.200 dòng
└── X_test.csv                  # 38.000 dòng (không có nhãn)
```

---

## Sử Dụng

Toàn bộ quy trình chạy trong notebook theo thứ tự:

```bash
jupyter lab
```

| Notebook | Mục đích |
|----------|---------|
| `01_eda.ipynb` | Phân tích phân phối dữ liệu, tương quan đặc trưng–mục tiêu, phát hiện bất thường đa chiều |
| `02_feature_engineering.ipynb` | Khám phá và kiểm tra FeaturePipeline |
| `03_modeling.ipynb` | Huấn luyện XGBoost → LightGBM → LSTM → Transformer → Ensemble, so sánh và sinh submission |

---

## Phân Tích EDA — `01_eda.ipynb`

Notebook EDA được tổ chức thành 6 phần chính, chỉ sử dụng `X_train` / `Y_train` (ngoại trừ biểu đồ so sánh độ dài chuỗi):

| # | Phần | Nội dung chính |
|---|------|----------------|
| 1 | Tải dữ liệu & Kiểm tra chất lượng | Shape, dtypes, missing values, trích xuất chuỗi |
| 2 | Phân tích chuỗi | Phân phối độ dài (train/val/test), từ vựng action, top-50 tần suất |
| 3 | Phân phối nhãn đích | Bar chart 6 targets, ma trận tương quan, kiểm tra class balance |
| 4 | Phân tích hành vi khách hàng | Entropy/uniqueness/length, tương quan đặc trưng–mục tiêu |
| 5 | Phát hiện bất thường | IQR (ngưỡng độ dài), Z-score đa chiều, hồ sơ outlier, Zipf / action hiếm |
| 6 | Kết luận & Insights nghiệp vụ | Tổng hợp insights, ý nghĩa cho mô hình, anomaly summary |

### Phát Hiện Bất Thường (Section 5) — Chi Tiết

Bổ sung 4 lớp phân tích bất thường:

| Phân tích | Phương pháp | Kết quả chính |
|-----------|------------|---------------|
| IQR theo độ dài chuỗi | Q1−1.5×IQR, Q3+1.5×IQR | 1.026 chuỗi quá dài (>24 actions, 2%) |
| Z-score đa chiều | Z > 3 trên length / entropy / uniqueness_ratio | 1.122 outliers tổng (2.2%) |
| So sánh hồ sơ outlier | Normalized bar chart + entropy nhãn đích | Outlier dài 2.5× nhưng phân phối nhãn không lệch |
| Phân tích Zipf / action hiếm | Log-log, độ phủ tích lũy | 3 dominant actions (18.9%), 110 rare actions (1.84%) |

### Biểu Đồ Sinh Ra (`outputs/figures/`)

| File | Nội dung |
|------|---------|
| `sequence_length_distribution.png` | Histogram độ dài chuỗi train/val/test |
| `action_frequency_distribution.png` | Top-50 tần suất action, phân phối log-scale |
| `target_distributions.png` | Bar chart phân phối 6 targets |
| `target_correlation.png` | Ma trận tương quan giữa 6 targets |
| `features_target_correlation.png` | Heatmap đặc trưng chuỗi vs mục tiêu |
| `iqr_anomaly_detection.png` | Histogram + box plot với ngưỡng IQR |
| `multidim_outlier_scatter.png` | Scatter 3 cặp chiều, tô màu loại bất thường |
| `outlier_profile_target_impact.png` | Hồ sơ so sánh + entropy nhãn normal vs outlier |
| `action_frequency_outliers.png` | Zipf curve, độ phủ tích lũy, phân phối action hiếm |

---

## Các Mô Hình

| # | Mô hình | Nhóm | Đầu vào | Đặc điểm nổi bật |
|---|---------|------|---------|-----------------|
| 1 | XGBoost | ML | FeaturePipeline | GPU hist; song song hóa 6 mục tiêu |
| 2 | LightGBM | ML | FeaturePipeline | GBDT nhanh; early stopping |
| 3 | LSTM | DL | SequencePreprocessor | BiLSTM + additive attention; AMP |
| 4 | Transformer | DL | SequencePreprocessor | CLS token + 4 encoder layers; AMP |
| 5 | Ensemble | Ensemble | Tất cả mô hình trên | Weighted soft-voting trên xác suất |

### Pipeline Đặc Trưng (ML — ~4.000 chiều)

| Bộ trích xuất | Số chiều | Mô tả |
|--------------|---------|-------|
| TF-IDF | 2.000 | N-gram (1–3) có trọng số TF-IDF |
| N-gram | 300 | Top-100 unigram / bigram / trigram |
| Statistical | 218 | Độ dài, entropy, tần suất chuyển trạng thái |
| Histogram | 1.524 | Đếm chính xác 254 action + last-5 positional one-hots |

### Ánh Xạ Vocab Dày (Dense Vocab Remapping)

ID hành động thô trải từ 102–24.438 nhưng chỉ có **254 giá trị duy nhất**.
`SequencePreprocessor` ánh xạ chúng về chỉ số dày 1–254 (0 = padding),
thu nhỏ bảng embedding **96 lần** và đảm bảo mọi slot đều nhận gradient.

---

## Cấu Hình

Toàn bộ siêu tham số trong `configs/config.yaml`:

```yaml
preprocessing:
  max_sequence_length: 64      # max thực tế là 37; thêm buffer an toàn
  truncation_strategy: "pre"   # giữ lại các hành động gần nhất

features:
  max_features: 2000           # TF-IDF top features

models:
  xgboost:
    device: "cuda"             # GPU hist (1.3 GB VRAM với 45.900 × 4.042)
    n_estimators: 400
    early_stopping_rounds: 50

  lightgbm:
    device: "gpu"
    n_estimators: 400

  lstm:
    hidden_dim: 512
    num_layers: 3
    bidirectional: true
    patience: 10

  transformer:
    d_model: 128
    nhead: 4
    num_layers: 4
    patience: 20
    warmup_epochs: 3
```

---

## Tái Tạo Kết Quả

Seed cố định ở 42 cho numpy, Python random và PyTorch:

```python
from src.utils import set_seed
set_seed(42)
```

### Kết Quả EDA Quan Trọng

| Insight | Giá trị |
|---------|---------|
| Outlier theo độ dài (IQR) | 1.026 chuỗi >24 actions (2.0%) |
| Outlier đa chiều (Z-score > 3) | 1.122 chuỗi (2.2%) |
| Action dominant (>3% tổng) | 3 actions — 105 (8.6%), 102 (6.3%), 103 (3.9%) |
| Action hiếm (<0.05% tổng) | 110 actions (1.84% tổng lượt tương tác) |
| Độ phủ 80% interaction | Top 59 actions đủ |
| Repetition tối đa | 2 (không có bot behavior) |
| Phân phối nhãn outlier vs normal | Tương đương — outlier không cần loại bỏ |

---

## Ràng Buộc Cuộc Thi

| Ràng buộc | Chi tiết |
|-----------|---------|
| Tách Train / Val | **KHÔNG BAO GIỜ gộp** — kể cả khi tạo submission cuối |
| Giới hạn LLM | Không dùng mô hình > 0,5B tham số |
| Early stopping | Chỉ dùng fold nội bộ từ X_train — Y_val không bao giờ được dùng |
| Kiểu dữ liệu đầu ra | Tất cả 6 thuộc tính phải là `uint16` |

---

## Bản Quyền

Chỉ dùng cho mục đích thi đấu DATAFLOW 2026.
