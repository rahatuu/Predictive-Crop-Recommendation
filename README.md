# 🌾 Predictive Modeling for Crop Recommendation Using Soil Nutrients and Climatic Parameters

A machine learning pipeline that recommends the most suitable crop for a plot of land based on 7 measurable soil and climatic parameters — nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.55%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Crop selection has significant implications for yield, resource use, and economic return, and is traditionally based on farmers' personal experience — which can be unreliable as soil and climatic conditions vary. This project builds and evaluates an automatic crop recommendation system using machine learning, trained on a labeled dataset of 22 crop types, to help identify the most suitable crop for a given location.

**Group 8 — CSE437 Course Project**

## 📊 Dataset

[Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

- 2,200 samples across 22 crop classes (100 samples per class, perfectly balanced)
- 7 numeric features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`
- No missing values

## 🔧 Pipeline

**1. Data Preparation**
- Load & validate (dtypes, nulls, class balance)
- Stratified 80/20 train-test split (fixed seed)
- Feature scaling with `StandardScaler`
- Label encoding with `LabelEncoder`

**2. Model Training**
Five classifiers trained and compared on the same scaled features:
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbors (k = 5)
- Logistic Regression (multinomial)

**3. Optimization & Deployment**
- Hyperparameter tuning via `GridSearchCV` (5-fold stratified cross-validation)
- Feature importance analysis
- Model, scaler, and label encoder saved with `joblib`
- Single inference function for real-time crop prediction

## 📈 Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---|---|---|---|
| Random Forest (baseline) | 99.55% | 99.57% | 99.55% | 99.55% |
| Gradient Boosting | 98.86% | 98.97% | 98.86% | 98.87% |
| SVM (RBF kernel) | 98.41% | 98.56% | 98.41% | 98.40% |
| K-Nearest Neighbors | 97.95% | 98.04% | 97.95% | 97.93% |
| Logistic Regression | 97.27% | 97.40% | 97.27% | 97.25% |
| **Random Forest (tuned)** | **99.55%** | **99.57%** | **99.55%** | **99.55%** |

**Feature importance (tuned Random Forest):** Rainfall (22.1%) and humidity (22.0%) were the strongest predictors, together accounting for ~44% of the model's decisions — followed by potassium (17.8%), phosphorus (14.9%), nitrogen (10.7%), temperature (7.5%), and pH (5.1%).

<p float="left">
  <img src="plots/model_comparison.png" width="48%" />
  <img src="plots/feature_importance.png" width="48%" />
</p>

## 📂 Repository Structure

```
crop-recommendation-ml/
├── README.md
├── Crop_Recommendation_Training.ipynb   # Full training pipeline (Colab-ready)
├── data/
│   └── Crop_recommendation.csv
├── models/
│   ├── crop_model.joblib                # Tuned Random Forest
│   ├── scaler.joblib                    # StandardScaler
│   └── label_encoder.joblib             # LabelEncoder
├── report/
│   └── CSE437_group_8.pdf               # Full project report
└── plots/
    ├── model_comparison.png
    ├── feature_importance.png
    └── confusion_matrix_RandomForest_Tuned.png
```

## 🚀 How to Run

**Option A — Google Colab (recommended)**
Open [`Crop_Recommendation_Training.ipynb`](Crop_Recommendation_Training.ipynb) in Colab and run all cells — the notebook will prompt you to upload the CSV.

**Option B — Local**
```bash
git clone https://github.com/<your-username>/crop-recommendation-ml.git
cd crop-recommendation-ml
pip install pandas scikit-learn matplotlib seaborn joblib xgboost
jupyter notebook Crop_Recommendation_Training.ipynb
```

**Inference on new data**
```python
import joblib
import pandas as pd

model = joblib.load("models/crop_model.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

sample = pd.DataFrame([[90, 42, 43, 20.9, 82.0, 6.5, 202.9]],
                       columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
prediction = model.predict(scaler.transform(sample))
print(label_encoder.inverse_transform(prediction)[0])  # -> predicted crop
```

## 🛠️ Tech Stack
Python · pandas · scikit-learn · matplotlib · seaborn · joblib

## 👥 Team — Group 8

| Name | Student ID |
|---|---|
| Saikat Rahman Asif | 24341132 |
| Nahian Abid | 21201690 |
| Mohammod Roby | 23141023 |
| Md. Rahatul Islam Rahat | 23301569 |

## 📄 Report

Full project report available at [`report/CSE437_group_8.pdf`](report/CSE437_group_8.pdf).
