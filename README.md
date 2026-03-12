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

![](images/Main1.png)

**Figure 1 — End-to-end project pipeline:**
*Raw Data → Target Cleaning → Feature Engineering → Imputation → Scaling/Encoding → Model Testing*

---

