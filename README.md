# DATAFLOW 2026 - User Behavior Prediction

Multi-output classification model to predict 6 independent behavioral attributes of customers based on their historical action sequences.

## Problem Overview

- **Input**: Variable-length sequences of encoded user actions (4 weeks of data)
- **Output**: 6 independent attributes (attr_1 to attr_6) in UINT16 format
- **Metric**: Exact-Match Accuracy (all 6 must be correct)

## Project Structure

```
user-behavior-prediction/
├── configs/
│   └── config.yaml          # Configuration file
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb    # Model training & evaluation
├── outputs/
│   ├── figures/             # EDA visualizations
│   ├── models/              # Saved models
│   └── submissions/         # Kaggle submissions
├── src/
│   ├── data/
│   │   ├── loader.py        # Data loading utilities
│   │   └── preprocessor.py  # Sequence preprocessing
│   ├── features/
│   │   ├── sequence_features.py   # TF-IDF, N-gram features
│   │   ├── statistical_features.py
│   │   └── feature_pipeline.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py
│   │   └── transformer_model.py
│   ├── evaluation/
│   │   └── metrics.py       # Exact-match accuracy, F1, etc.
│   └── utils/
│       ├── seed.py          # Reproducibility
│       └── helpers.py       # Config loading, logging
├── train.py                 # Main training script
├── predict.py               # Generate submissions
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

Place data files in `../data/` directory:
- `X_train.csv`, `Y_train.csv`
- `X_val.csv`, `Y_val.csv`
- `X_test.csv`

## Usage

### 1. Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train Models
```bash
# Train all models
python train.py

# Train specific models
python train.py --models xgboost lstm transformer
```

### 3. Generate Submission
```bash
python predict.py --model xgboost --team_name your_team
```

## Models Implemented

| Model | Architecture | Input |
|-------|-------------|-------|
| XGBoost | Gradient Boosting | TF-IDF + Statistical features |
| LSTM | Bidirectional LSTM | Padded sequences |
| Transformer | Encoder-only Transformer | Padded sequences |

## Evaluation Metrics

- **Primary**: Exact-Match Accuracy (competition metric)
- **Auxiliary**:
  - Per-attribute accuracy
  - Macro F1-Score
  - Weighted F1-Score

## Configuration

Edit `configs/config.yaml` to modify:
- Preprocessing parameters (max sequence length, padding)
- Model hyperparameters
- Training settings (seed, GPU usage)

## Reproducibility

All random operations use fixed seeds:
```python
from src.utils import set_seed
set_seed(42)
```

## Competition Rules

1. **EDA Phase**: Keep Train and Validation separate
2. **Final Submission**: Can merge Train + Validation
3. **No LLMs > 0.5B parameters**

## Team

[Your Team Name]

## License

For DATAFLOW 2026 competition use only.
