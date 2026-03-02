# Project Requirements — DataFlow 2026: User Behavior Prediction (Bán kết)

## 1. Problem Statement

Dự đoán hành vi của người dùng dựa trên lịch sử chuỗi hành động. Với mỗi người dùng, cho trước
một chuỗi các hành động (variable-length), mô hình phải dự đoán đồng thời 6 thuộc tính mục tiêu.

| Thuộc tính | Kiểu dữ liệu | Số lớp |
|-----------|-------------|--------|
| `attr_1`  | uint16      | 12     |
| `attr_2`  | uint16      | 31     |
| `attr_3`  | uint16      | 99     |
| `attr_4`  | uint16      | 12     |
| `attr_5`  | uint16      | 31     |
| `attr_6`  | uint16      | 99     |

**Primary metric:** Exact-Match Accuracy
> Một dự đoán được tính đúng khi và CHỈ KHI cả 6 thuộc tính đều đúng.

$$\text{Exact-Match Accuracy} = \frac{\text{Số mẫu dự đoán đúng toàn bộ 6 thuộc tính}}{N}$$

---

## 2. Data Layout

```
data/
├── X_train.csv     # Chuỗi hành động — tập huấn luyện
├── Y_train.csv     # Nhãn 6 thuộc tính — tập huấn luyện
├── X_val.csv       # Chuỗi hành động — tập kiểm tra nội bộ
├── Y_val.csv       # Nhãn 6 thuộc tính — tập kiểm tra nội bộ
└── X_test.csv      # Chuỗi hành động — tập thi đấu (không có nhãn)
```

### Định dạng X (input)

| Cột         | Mô tả                                        |
|-------------|----------------------------------------------|
| `id`        | Định danh người dùng (string)                |
| `feature_1` … `feature_37` | Chỉ số hành động (integer), phần đuôi là NaN nếu chuỗi ngắn hơn 37 |

Số cột feature tối đa trong dữ liệu cuộc thi: **37** (`feature_1`–`feature_37`).

### Định dạng Y (target)

| Cột       | Mô tả                       |
|-----------|------------------------------|
| `id`      | Định danh người dùng         |
| `attr_1`–`attr_6` | Nhãn phân loại (uint16) |

### Thống kê dữ liệu

```
X_train: 51,000 rows × 37 action columns
Sequence lengths: 3–37 (mean ~13)
Unique action IDs: 254  (sparse range: 102–24438)
```

---

## 3. Quy tắc Tách Train / Validation — BẮT BUỘC TUYỆT ĐỐI

> **NGHIÊM CẤM TUYỆT ĐỐI** kết hợp tập Train và Validation dưới bất kỳ hình thức nào.
> Mọi cờ `--final`, hàm `merge_train_val()`, hoặc bất kỳ cơ chế nào gộp nhãn Y_val vào
> quá trình huấn luyện đều **vi phạm quy tắc này và phải bị loại bỏ**.

**Luồng dữ liệu phải tuân theo quy tắc:**

```
X_train + Y_train  ──►  Huấn luyện mô hình
                             │
X_val   (features only) ──►  Dự đoán sau huấn luyện
                             │
Y_val   (labels)    ──►  So sánh kết quả (đánh giá offline)
                             │
X_test  (features only) ──►  Sinh file nộp bài (không có nhãn)
```

**Quy tắc nghiêm ngặt:**

- `X_val` / `Y_val` **KHÔNG được dùng làm nguồn nhãn trong quá trình huấn luyện** — không có ngoại lệ.
- `Y_val` **KHÔNG được dùng cho early stopping**. Early stopping chỉ được thực hiện qua
  cross-validation nội bộ trên tập train (tách một phần nhỏ của X_train ra làm early-stop fold).
- `X_test` / các nhãn tương ứng **KHÔNG bao giờ được nhìn thấy** trong bước huấn luyện.
- `merge_train_val()` **KHÔNG được gọi trong bất kỳ bước nào**, kể cả khi tạo submission cuối.
- Submission cuối được tạo ra bằng cách predict trực tiếp từ mô hình đã train trên X_train.
- Mọi báo cáo kết quả (accuracy, F1…) phải được tính trên `X_val / Y_val` hoặc cross-validation
  trên tập train, **không phải trên tập train**.

---

## 4. Luồng xử lý chuẩn

```
1. Load     X_train, Y_train, X_val, Y_val
2. Fit      SequencePreprocessor   trên  train_sequences  (KHÔNG dùng val/test)
3. Fit      FeaturePipeline        trên  train_sequences  (KHÔNG dùng val/test)
4. Fit      TargetEncoder          trên  Y_train          (KHÔNG dùng val/test)
5. Transform val_sequences và Y_val với các preprocessor đã fit ở bước trên
6. Train    model(s) với X_train
            - Early stopping: chia nội bộ X_train thành 90% train / 10% early-stop fold
            - KHÔNG dùng X_val để làm tín hiệu early stopping
7. Evaluate → so sánh y_pred(X_val) vs Y_val → báo cáo metrics trên val set
8. [Submission] predict X_test với mô hình đã train → xuất file nộp bài
```

---

## 5. Mô hình sử dụng

Ba nhóm mô hình được triển khai song song:

### 5.1 Machine Learning (ML)

| Mô hình  | Input                          | Đặc điểm                                         |
|----------|-------------------------------|--------------------------------------------------|
| XGBoost  | `FeaturePipeline.transform()` | TF-IDF + statistical + histogram features, GPU  |
| LightGBM | `FeaturePipeline.transform()` | Gradient boosting nhanh hơn, GPU optional        |

Feature vector cho ML bao gồm:
- **TF-IDF** (ngram 1–3, max 5000 features): biểu diễn bag-of-n-grams trên chuỗi hành động
- **Statistical features**: length, mean, std, min, max, unique count
- **Transition features**: tần suất cặp hành động liên tiếp
- **Histogram features**: đếm chính xác số lần xuất hiện mỗi action ID (254 bins) + last-K positional one-hots

### 5.2 Deep Learning (DL)

| Mô hình | Input                                | Đặc điểm                                        |
|---------|-------------------------------------|-------------------------------------------------|
| LSTM    | `SequencePreprocessor.transform()` | Bidirectional + Attention Pooling, hidden=512   |

Kiến trúc LSTM:
- Dense vocab remapping: 254 unique tokens → embedding 255×128 (0 = padding)
- Bidirectional LSTM (3 layers, hidden_dim=512)
- Additive attention pooling (toàn bộ timestep, không chỉ last hidden state)
- 6 output heads độc lập, mỗi head cho một thuộc tính
- Label smoothing 0.05, AdamW + LambdaLR (warmup 10% + cosine decay)
- Mixed precision (AMP) khi dùng GPU

### 5.3 Transformer

| Mô hình     | Input                                | Đặc điểm                                        |
|-------------|-------------------------------------|-------------------------------------------------|
| Transformer | `SequencePreprocessor.transform()` | CLS token + 4 encoder layers, d_model=128       |

Kiến trúc Transformer:
- Dense vocab embedding 255×128 + Positional Encoding
- CLS token prepended, 4 TransformerEncoderLayer (nhead=4, dim_ff=512)
- CLS output dùng cho 6 output heads (Linear → ReLU → Dropout → Linear)
- Label smoothing 0.05, AdamW + LambdaLR (warmup 3 epochs + cosine decay)
- Mixed precision (AMP) khi dùng GPU

### 5.4 Ensemble

| Mô hình  | Input                        | Đặc điểm                                        |
|----------|-----------------------------|-------------------------------------------------|
| Ensemble | Inputs từ tất cả mô hình    | Weighted soft-voting trên xác suất từng class   |

- `EnsembleMultiOutput`: nhận danh sách model đã fit + input riêng cho từng model
- Weights được calibrate từ per-attribute accuracy trên val set
- `predict_from_inputs([X_seq, X_seq, X_feat])` cho LSTM + Transformer + XGBoost

---

## 6. Dense Vocab Remapping (Tối ưu quan trọng)

Raw action IDs trải rộng từ 102 đến 24438 nhưng chỉ có **254 giá trị unique**.
`SequencePreprocessor` ánh xạ chúng sang indices dày đặc 1–254 (0 = padding),
giảm kích thước embedding table từ 24,439 xuống còn 255 (giảm 96 lần).
Mọi slot embedding đều nhận gradient → DL models huấn luyện hiệu quả hơn nhiều.

**Quan trọng:** vocab mapping được fit CHỈ trên `train_sequences`. Các token lạ ở val/test
sẽ được ánh xạ về 0 (padding) — không gây data leakage.

---

## 7. Ràng buộc cuộc thi

| Ràng buộc            | Chi tiết                                             |
|----------------------|------------------------------------------------------|
| Mô hình ngôn ngữ lớn | Không được dùng LLM có hơn **0.5B tham số**         |
| Merge train+val      | **NGHIÊM CẤM** — không được gộp trong bất kỳ bước nào |
| Kiểu dữ liệu output  | Tất cả 6 thuộc tính phải xuất ra dạng `uint16`       |
| Seed                 | Cố định seed (`seed: 42`) để tái tạo kết quả         |
| Early stopping       | Chỉ dùng internal fold tách từ X_train, không dùng Y_val |

---

## 8. Yêu cầu kỹ thuật (Technical Requirements)

### 8.1 Môi trường

```
Python  >= 3.10
CUDA    >= 11.8  (tùy chọn, cho GPU training)
RAM     >= 16 GB
```

### 8.2 Python Dependencies

```
# ML / DL core
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
xgboost >= 2.0
lightgbm >= 4.0
torch >= 2.0

# Feature engineering
scipy >= 1.10

# Utilities
tqdm >= 4.65
joblib >= 1.3
pyyaml >= 6.0

# Visualization (notebooks)
matplotlib >= 3.7
seaborn >= 0.12
plotly >= 5.15

# Notebook
jupyter >= 1.0
ipykernel >= 6.25
```

### 8.3 Cấu trúc thư mục

```
user-behavior-prediction/
├── configs/config.yaml          # Tất cả hyperparameter (cấu hình duy nhất)
├── data/                        # Dữ liệu gốc (không commit lên repo)
├── outputs/
│   ├── models/                  # Serialized models + preprocessors (.pkl)
│   ├── submissions/             # File nộp bài (.csv)
│   └── figures/                 # Biểu đồ EDA / kết quả
├── src/
│   ├── data/                    # DataLoader, SequencePreprocessor, TargetEncoder
│   ├── features/                # FeaturePipeline (TF-IDF, thống kê, n-gram, histogram)
│   ├── models/                  # BaseModel + XGBoost / LightGBM / LSTM / Transformer / Ensemble
│   ├── evaluation/              # Metrics (exact_match, per_attribute, F1)
│   └── utils/                   # load_config, set_seed, logger, pickle helpers
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis (chỉ dùng X_train/Y_train)
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── train.py                     # CLI huấn luyện (dev mode: train trên X_train, eval trên X_val)
├── train_cv.py                  # CLI cross-validation (ước tính score đáng tin cậy hơn)
├── predict.py                   # CLI dự đoán / tạo submission từ X_test
├── requirements.txt
└── CLAUDE.md
```

---

## 9. CLI Interface

### train.py

```bash
# Huấn luyện tất cả mô hình và đánh giá trên tập val
python train.py

# Huấn luyện mô hình cụ thể
python train.py --models xgboost
python train.py --models xgboost lstm transformer

# Dùng config khác
python train.py --config configs/config.yaml
```

**Lưu ý:** `train.py` KHÔNG có cờ `--final` và KHÔNG gọi `merge_train_val()`.
Mô hình luôn được train trên X_train và evaluate trên X_val (held-out).

**Output bắt buộc sau khi train:**

```
[Model Comparison — Val Set]
xgboost     exact_match=0.XXXX   macro_f1=0.XXXX
lightgbm    exact_match=0.XXXX   macro_f1=0.XXXX
lstm        exact_match=0.XXXX   macro_f1=0.XXXX
transformer exact_match=0.XXXX   macro_f1=0.XXXX
ensemble    exact_match=0.XXXX   macro_f1=0.XXXX
```

### train_cv.py (Khuyến nghị cho ước tính score)

```bash
# K-Fold CV trên tập train — không dùng Y_val (đáng tin cậy hơn)
python train_cv.py --models xgboost --n_folds 5
python train_cv.py --models xgboost lstm --n_folds 5
```

### predict.py

```bash
# Tạo submission từ X_test (chỉ dùng sau khi train xong)
python predict.py --model xgboost --team_name ten_nhom
python predict.py --model ensemble --team_name ten_nhom
```

---

## 10. Tiêu chí đánh giá kết quả (Evaluation Criteria)

| Metric                  | Mô tả                                     | Ưu tiên |
|-------------------------|-------------------------------------------|---------|
| Exact-Match Accuracy    | Tất cả 6 thuộc tính đúng                 | **Chính** |
| Per-attribute Accuracy  | Độ chính xác từng thuộc tính riêng lẻ    | Phụ     |
| Macro F1                | F1 trung bình các class của mỗi thuộc tính | Phụ   |
| Weighted F1             | F1 có trọng số theo tần suất class        | Phụ     |

---

## 11. Kiểm tra trước khi nộp bài (Pre-submission Checklist)

- [ ] Tất cả metrics được tính trên `X_val / Y_val`, **không phải train set**
- [ ] **`merge_train_val()` KHÔNG được gọi** trong bất kỳ bước nào
- [ ] **`--final` mode KHÔNG tồn tại** trong train.py
- [ ] `Y_val` không được dùng làm early stopping signal
- [ ] Early stopping chỉ dùng internal fold tách từ X_train
- [ ] File submission có đúng định dạng: `id, attr_1, attr_2, attr_3, attr_4, attr_5, attr_6`
- [ ] Tất cả 6 cột là kiểu `uint16`
- [ ] Seed đã được cố định (`seed: 42`)
- [ ] Preprocessors (SequencePreprocessor, FeaturePipeline, TargetEncoder) fit CHỈ trên train
- [ ] Dense vocab mapping fit CHỈ trên train_sequences
- [ ] Kết quả có thể tái tạo (reproducible)
- [ ] Ba nhóm mô hình được huấn luyện: ML (XGBoost/LightGBM), DL (LSTM), Transformer
