# Car Anakysis

# Used Cars Price Prediction — Model Benchmark (5-Fold Cross-Validation)

This repository compares several regression models for **predicting used car prices** using a unified pipeline with feature engineering, preprocessing, and cross-validation.

The goal is to obtain a **fair comparison of models under identical conditions** and select the best one based on prediction error in **real price (£)**.

---

# Pipeline Overview

**Target variable**

Models are trained on the logarithm of the price to stabilize variance and reduce the influence of extreme values.

---

# Feature Engineering

All feature transformations are handled by:

Examples of engineered features include:

- `CAE` — compressed vehicle age
- `log_mileage` — logarithm of mileage
- `km_per_year`
- `mileage_age_interaction`
- binary indicators:
  - `high_mileage`
  - `large_engine`
  - `auto_large_engine`

These features capture **vehicle depreciation dynamics** and nonlinear relationships between mileage, age, and engine size.

---

# Input Features

### Numerical features

- `CAE`
- `log_mileage`
- `km_per_year`
- `engineSize`
- `mpg`
- `tax`
- `mileage_age_interaction`
- `high_mileage`
- `large_engine`
- `auto_large_engine`

### Categorical features

- `brand`
- `model`
- `transmission`
- `fuelType`


Each model is evaluated across **5 folds**, producing mean and standard deviation metrics.

---

# Models Compared

The following models are benchmarked:

- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

For fairness, CatBoost is also evaluated using the same **OneHotEncoder preprocessing** as the other models.

---

# Evaluation Metrics

Metrics are computed on both **log scale** and **real price scale**.

### Log scale metrics

- `RMSE_log`
- `MAE_log`
- `R²_log`

### Real price metrics (£)

Predictions are transformed back using:

Then evaluated with:

- `RMSE_real`
- `MAE_real`
- `R²_real`

Real-price metrics are easier to interpret in business terms.

---

# Cross-Validation Results

| Model | RMSE_log | MAE_log | R²_log | RMSE_real (£) | MAE_real (£) | R²_real |
|------|---------|--------|-------|---------------|-------------|--------|
| LightGBM | 0.0936 | 0.0662 | 0.9690 | **1841** | **1091** | **0.964** |
| XGBoost | 0.0960 | 0.0693 | 0.9674 | 1908 | 1152 | 0.962 |
| RandomForest | 0.1020 | 0.0699 | 0.9632 | 1929 | 1135 | 0.961 |
| CatBoost | 0.0960 | 0.0698 | 0.9674 | 1930 | 1167 | 0.961 |
| DecisionTree | 0.1298 | 0.0860 | 0.9404 | 2399 | 1388 | 0.939 |

---

# Best Model

**LightGBM** achieved the best performance:
