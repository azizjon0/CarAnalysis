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



# Stress Test Evaluation (Group Split)

To evaluate how well the model generalizes to **unseen car groups**, a stress test was performed using a **GroupShuffleSplit** strategy.

Instead of random splits, vehicles were grouped by:



This ensures that **specific brand–model combinations appearing in training are not present in the test set**, simulating a more realistic scenario where the model encounters unseen vehicle types.

---

# Why This Stress Test Matters

Standard K-Fold validation may produce overly optimistic metrics because:

- identical or very similar cars can appear in both train and validation folds
- models can implicitly memorize price patterns

Group-based splitting forces the model to **generalize to unseen vehicle groups**, making the evaluation significantly more realistic.

---

# Dataset Split

| Metric | Value |
|------|------|
| Training samples | 84,677 |
| Test samples | 21,590 |
| Training groups | 157 |
| Test groups | 40 |

Each group represents a unique **brand–model combination**.

---

# Stress Test Results

| Model | R² (log) | RMSE (log) | MAE (log) | R² (£) | RMSE (£) | MAE (£) |
|------|---------|-----------|----------|--------|----------|---------|
| CatBoost | **0.836** | 0.226 | 0.179 | **0.849** | **3663** | **2614** |
| XGBoost | 0.825 | 0.234 | 0.183 | 0.833 | 3861 | 2692 |
| LightGBM | 0.823 | 0.235 | 0.185 | 0.814 | 4065 | 2750 |
| RandomForest | 0.825 | 0.234 | 0.186 | 0.813 | 4080 | 2749 |
| DecisionTree | 0.762 | 0.272 | 0.214 | 0.746 | 4752 | 3172 |

---

# Key Observations

### 1️⃣ Performance drops compared to standard CV

In regular cross-validation the best model achieved:
R² ≈ 0.96
RMSE ≈ £1841


This drop is expected because the model must now **predict prices for unseen brand–model combinations**.

---

### 2️⃣ CatBoost generalizes best

CatBoost achieved the best performance in the stress test:
R²_real ≈ 0.849
RMSE_real ≈ £3663


This suggests it handles **categorical relationships and unseen groups better** than the other models.

---

### 3️⃣ Tree ensembles still perform well

Even under strict evaluation:

- Gradient boosting models remain strongest
- Single decision trees degrade significantly

---

# Interpretation

This stress test shows that the model:

- learns **general price patterns**
- not just memorization of specific vehicles

However, predicting prices for **completely unseen vehicle groups** remains significantly harder.

---

# Takeaway

Two evaluation perspectives are important:

| Evaluation | Purpose |
|------|------|
| Standard Cross Validation | Overall predictive performance |
| Group Stress Test | Real-world generalization |

Using both provides a **more honest estimate of model robustness**.

---

# Next Improvements

Possible directions for improving generalization:

- add **brand-level statistical features**
- introduce **hierarchical encoding**
- use **target encoding for categorical features**
- increase dataset diversity
- incorporate **market-level signals**

---

