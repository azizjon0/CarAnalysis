# Car Analysis 

## 🛠 Tech Stack
| Category | Tools |
|--------|--------|
| Language | Python 3.10+ |
| ML Model | LightGBM |
| ML Framework | scikit-learn |
| Explainability | SHAP |
| App Framework | Streamlit |
| Data Processing | Pandas, NumPy |

---

## 📋 Project Overview

This project develops a **machine learning pipeline to predict used car resale prices in the UK market**.

The system is trained on a dataset of **~100,000 car listings from Kaggle** and evaluates multiple machine learning models under the same preprocessing and validation conditions.

After benchmarking **five models**, **LightGBM** achieved the best performance:

- **R² Score:** `0.9643`

The final system integrates **SHAP explainability** to make predictions transparent and interpretable.

A **Streamlit prototype application** allows users to:

- Input vehicle attributes  
- Receive an estimated resale price  
- View **feature-level explanations** showing which variables influenced the prediction

---

## 🎯 Research Contributions

This project addresses three gaps identified in existing literature:

- **Systematic multi-model benchmarking**  
  Models are evaluated under identical preprocessing and validation pipelines.

- **Explainability as a first-class design objective**  
  SHAP explanations are integrated directly into the pipeline rather than added post-hoc.

- **Interactive ML prototype**  
  A functional **Streamlit interface** provides real-time predictions with interpretable explanations.


  
