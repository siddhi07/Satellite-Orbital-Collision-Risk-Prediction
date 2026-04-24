# Satellite Orbital Collision Risk Prediction

## Overview

This project builds an end-to-end machine learning pipeline for predicting **satellite orbital collision risk** using the Collision Avoidance Challenge dataset.

The original continuous `risk` score was converted into a **3-class classification problem**:
- Low Risk
- Medium Risk
- High Risk

The pipeline includes:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Comparison
- Final Prediction Generation

---

## Problem Statement

As the number of satellites and debris objects increases, close approaches between space objects are becoming more common. These conjunction events must be evaluated quickly to determine whether they are low priority or require urgent collision avoidance action.

Although the original target was continuous (`risk`), we chose a classification approach because real-world operational decisions are threshold-based.

---

## Dataset

**Source:** https://www.kaggle.com/datasets/shadmanrohan/collisionavoidancechallenge

### Final Dataset Used
- Original Train Records: **162,634**
- Original Test Records: **24,484**

### Final Split

Using `GroupShuffleSplit(event_id)`:
- Train: **130,596**
- Validation: **32,038**
- Test: **24,484**

### Features
- Original Features: **103**
- Final Engineered Features: **~145**

---

## Repository Structure

```
Satellite-Orbital-Collision-Risk-Prediction/
│
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── featured/
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_and_comparison.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── models/
│   └── best_xgboost_model.pkl
└── report/
    └── final_report.pdf
```

---

## Pipeline Workflow

Run notebooks in the following order.

### Step 1: EDA + Preprocessing

| Item | Details |
|------|---------|
| **Notebook** | `notebooks/01_eda_preprocessing.ipynb` |
| **Input** | `data/raw/train_data.csv` <br> `data/raw/test_data.csv` |
| **Tasks** | • Load raw datasets <br> • Missing value analysis <br> • Risk distribution analysis <br> • Group split using event_id <br> • Imputation using train statistics only <br> • Label generation using train quantiles <br> • Preprocessing / scaling |
| **Output** | `data/processed/train_processed.csv` <br> `data/processed/val_processed.csv` <br> `data/processed/test_processed.csv` <br> `data/featured/label_thresholds.json` |

### Step 2: Feature Engineering

| Item | Details |
|------|---------|
| **Notebook** | `notebooks/02_feature_engineering.ipynb` |
| **Input** | `data/processed/train_processed.csv` <br> `data/processed/val_processed.csv` <br> `data/processed/test_processed.csv` |
| **Tasks** | Created ~40 new features: <br> • Relative geometry <br> • Orbital deltas <br> • Uncertainty ratios <br> • Observation quality <br> • Time interactions <br> • Solar weather interactions |
| **Output** | `data/featured/train_featured.csv` <br> `data/featured/val_featured.csv` <br> `data/featured/test_featured.csv` |

### Step 3: Modeling + Comparison

| Item | Details |
|------|---------|
| **Notebook** | `notebooks/03_modeling_and_comparison.ipynb` |
| **Input** | `data/featured/train_featured.csv` <br> `data/featured/val_featured.csv` <br> `data/featured/test_featured.csv` |
| **Tasks** | • Remove leakage proxy columns <br> • Fix extreme values (cov_det_ratio) <br> • Train 4 models <br> • Evaluate on validation set <br> • Compare results <br> • Generate final predictions |
| **Models Used** | • Logistic Regression <br> • Random Forest <br> • XGBoost <br> • Deep Neural Network (PyTorch) |
| **Output** | `outputs/metrics/model_comparison_results.csv` <br> `outputs/predictions/test_predictions.csv` <br> `models/best_xgboost_model.pkl` |

---

## Configuration

### Common
```
seed = 42
metric = Macro F1
```

### Logistic Regression
```
max_iter = 2000
class_weight = balanced
```

### Random Forest
```
n_estimators = 300
max_depth = 12
```

### XGBoost
```
n_estimators = 300
max_depth = 6
learning_rate = 0.05
```

### Deep Neural Network (PyTorch)
```
epochs = 20
batch_size = 256
learning_rate = 0.0005
hidden_layers = [256, 128, 64]
```

### Hyperparameter Insights
- Logistic Regression needed higher iterations for convergence.
- Random Forest improved with more trees.
- XGBoost performed best with moderate depth and lower learning rate.
- DNN was initially unstable at learning rate 0.001.
- StandardScaler + learning rate 0.0005 significantly improved DNN.
- DNN improved from 3 epochs → 5 epochs → 20 epochs.

---

## Evaluation Metrics

Models were evaluated using:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

**Primary metric:** Macro F1

---

## Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **XGBoost** | 0.8963 | 0.8809 |
| **DNN** | 0.8619 | 0.8427 |
| **Random Forest** | 0.7903 | 0.7631 |
| **Logistic Regression** | 0.4988 | 0.2219 |

### Best Model
**XGBoost**

---

## Reproducibility

This project supports reproducibility through:
- Fixed random seed
- Centralized configuration
- Group split using event_id
- Same metrics across all models
- Saved intermediate datasets
- Structured notebook pipeline

---

## Installation

```bash
git clone https://github.com/siddhi07/Satellite-Orbital-Collision-Risk-Prediction.git
cd Satellite-Orbital-Collision-Risk-Prediction
pip install -r requirements.txt
```

---

## Data Setup

Download the Kaggle dataset and place files inside `data/raw/`

**Required files:**
- `train_data.csv`
- `test_data.csv`

---

## Run Project

```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling_and_comparison.ipynb
```

---

## Deployment Perspective

The final XGBoost model can be deployed as:
- Risk prediction API
- Monitoring dashboard
- High-risk alert system
- Scheduled retraining pipeline

---

## Future Improvements

- Sequence modeling across repeated events
- SHAP explainability
- Transformer / Graph models
- Real-time deployment

---

## Contributors

- **Siddhi Nirmale**
- **Diya Kaswa**
- **Adit Vakul**
