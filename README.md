## 🚗 Used Car Price Prediction (UK)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-9ACD32?logo=lightgbm\&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Framework-F7931E?logo=scikit-learn\&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6F00)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy\&logoColor=white)

---

## 🛠 Tech Stack

| Category           | Tools         |
| ------------------ | ------------- |
| 🐍 Language        | Python 3.10+  |
| 🤖 ML Model        | LightGBM      |
| 📚 ML Framework    | scikit-learn  |
| 🔍 Explainability  | SHAP          |
| 🌐 App Framework   | Streamlit     |
| 📊 Data Processing | Pandas, NumPy |

---

## 📋 Project Overview

This project develops a **machine learning pipeline to predict used car resale prices in the UK market**.

The system is trained on a dataset of **~100,000 car listings from Kaggle** and benchmarks multiple machine learning models using the same preprocessing and validation pipeline.

After evaluating **five models**, **LightGBM** achieved the best performance:

**📈 R² Score:** `0.9643`

To ensure transparency, the system integrates **SHAP explainability**, allowing users to understand how each feature influences the predicted price.

A **Streamlit prototype application** enables users to:

* 🚘 Input vehicle attributes
* 💰 Receive an estimated resale price
* 📊 View **feature-level explanations** showing how each variable contributed to the prediction

---

## 🎯 Research Contributions

This project addresses three key gaps identified in existing literature:

### 📊 Systematic Multi-Model Benchmarking

All models are evaluated under identical preprocessing, training, and validation pipelines to ensure fair comparison.

### 🔎 Explainability by Design

Explainability is integrated using **SHAP** as a **core component of the model pipeline**, not added as a post-hoc feature.

### 🖥 Interactive ML Prototype

A functional **Streamlit application** allows real-time predictions with interpretable explanations, demonstrating practical deployment of the model.

---

## 🎯 Project Aims & Objectives

### 🎯 Primary Aim

To **build, evaluate, and deploy an accurate and interpretable machine learning model** for predicting **UK used car prices**, demonstrating that **predictive accuracy and explainability can be jointly optimised**.

---

### 📌 Objectives

* 📂 **Data Integration**
  Collect and integrate **~100,000 used car listings** from multiple manufacturer CSV files into a unified dataset.

* 🔎 **Exploratory Data Analysis (EDA)**
  Perform comprehensive EDA to identify **distribution patterns, outliers, and feature relationships**.

* 🧠 **Feature Engineering**
  Engineer domain-relevant variables including:

  * Vehicle age
  * Log-transformed mileage
  * Feature interaction terms
  * Binary market-segment indicators

* 🤖 **Model Training & Validation**
  Train and cross-validate five machine learning models:

  * LightGBM
  * XGBoost
  * CatBoost
  * Random Forest
  * Decision Tree

* 📊 **Model Selection**
  Select the best-performing model using evaluation metrics:

  | Metric | Purpose                  |
  | ------ | ------------------------ |
  | R²     | Variance explained       |
  | RMSE   | Error magnitude          |
  | MAE    | Average prediction error |

  Evaluation is performed on both **log-price and real-price scales**.

* 🔍 **Model Explainability**
  Compute **SHAP values** to generate:

  * Global explanations (**beeswarm plots**)
  * Local explanations (**waterfall plots**)

* 🧪 **Out-of-Distribution Stress Testing**
  Apply **GroupShuffleSplit by `brand_model`** to evaluate generalisation to unseen vehicle types.

* 🌐 **Prototype Deployment**
  Deploy a **Streamlit web application** that returns:

  * Predicted resale price
  * SHAP waterfall explanation of the prediction

---

## ⚙️ Project Pipeline

The workflow follows a **structured six-stage machine learning pipeline**, progressing from raw data ingestion to final model evaluation.

![](Images/Main1.png)

**Figure 1 — End-to-end project pipeline:**
*Raw Data → Target Cleaning → Feature Engineering → Imputation → Scaling/Encoding → Model Testing*

---

---

# 📊 Dataset & Feature Engineering

## 📂 Dataset

| Attribute              | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| 📍 **Source**          | Kaggle — Used Car Dataset (Ford & Mercedes + multi-manufacturer) |
| 📊 **Size**            | ~100,000 observations                                            |
| 🧾 **Raw Variables**   | 10                                                               |
| 🚘 **Brands**          | Audi, BMW, Ford, Mercedes, Toyota, Vauxhall, VW + others         |
| 🎯 **Target Variable** | `price (£)`                                                      |

To stabilise variance and improve model learning, the target variable was **log-transformed**:

```python
log_price = log1p(price)
```

---

## 🧠 Engineered Features

Domain-informed features were created to better capture **vehicle depreciation patterns and usage intensity**.

```python
vehicle_age    = current_year - year
CAE            = log1p(vehicle_age)           # Compressed Age Effect
log_mileage    = log1p(mileage)
km_per_year    = mileage / (vehicle_age + 1)  # Usage intensity
mileage_age_interaction = log_mileage * CAE   # Combined depreciation
high_mileage   = 1 if mileage > 100000
large_engine   = 1 if engineSize >= 2.0
auto_large_eng = 1 if Automatic & engineSize >= 2.0
```

These transformations help capture **nonlinear depreciation effects and interactions between vehicle age, mileage, and engine size**.

---

## 🔗 Feature Correlation Structure

**Figure 2 — Feature correlation heatmap**

![](Images/Frame9.png)

---

**Figure 3 — Pearson correlation matrix**

![](Images/Main2.png)

---

# 🔧 Model Pipeline Architecture

The **scikit-learn `Pipeline`** encapsulates all preprocessing within the model to **prevent data leakage during cross-validation**.

Separate preprocessing branches are used for **numerical and categorical features** before feeding into the final model.

**Figure 4 — Model pipeline structure**

![](Images/Frame11.png)

This architecture ensures **reproducible preprocessing across training and validation folds**.

---

## ⚙️ LightGBM Hyperparameters

```python
LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)
```

These parameters were tuned to balance **model complexity, generalisation performance, and training stability**.

---

# 📈 Results

## 5-Fold Cross-Validation Performance

| Model          | R² (Real)  | RMSE (£)   | MAE (£)    | R² (Log)   |
| -------------- | ---------- | ---------- | ---------- | ---------- |
| ✅ **LightGBM** | **0.9643** | **£1,841** | **£1,092** | **0.9690** |
| XGBoost        | 0.9616     | £1,908     | £1,153     | 0.9674     |
| CatBoost       | 0.9607     | £1,930     | £1,167     | 0.9674     |
| Random Forest  | 0.9608     | £1,930     | £1,135     | 0.9632     |
| Decision Tree  | 0.9393     | £2,400     | £1,388     | 0.9404     |

🏆 **LightGBM achieved the best overall performance**, delivering the highest R² and lowest prediction errors across evaluation metrics.

---

