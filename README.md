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
| `01_eda.ipynb` | Phân tích phân phối dữ liệu, tương quan đặc trưng–mục tiêu |
| `02_feature_engineering.ipynb` | Khám phá và kiểm tra FeaturePipeline |
| `03_modeling.ipynb` | Huấn luyện XGBoost → LightGBM → LSTM → Transformer → Ensemble, so sánh và sinh submission |

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
