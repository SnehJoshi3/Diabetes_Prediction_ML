# 🩺 PIMA Indians Diabetes Classification using K-Nearest Neighbors (KNN)

An end-to-end Machine Learning pipeline and Exploratory Data Analysis (EDA) on the PIMA Indians Diabetes dataset. This project cleans, visualizes, scales, and evaluates a **K-Nearest Neighbors (KNN)** classification model to predict diabetic outcomes based on diagnostic measurements.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Key Features & Exploratory Data Analysis](#-key-features--exploratory-data-analysis)
- [Preprocessing & Model Architecture](#-preprocessing--model-architecture)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Installation & Usage](#-installation--usage)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview
The goal of this project is to build a binary classification model that accurately predicts whether a patient has diabetes (`Outcome = 1`) or not (`Outcome = 0`) using clinical parameters. 

**Highlights:**
* Detailed Exploratory Data Analysis (EDA) using distribution plots, pair plots, and box plots.
* Standardization of numerical features using `StandardScaler`.
* Hyperparameter tuning across multiple $K$ values to determine optimal neighbor distance.
* Evaluation via Confusion Matrix and Classification Report metrics (Precision, Recall, F1-Score).

---

## 📊 Dataset Information
The dataset used in this project is the **PIMA Indians Diabetes Database** sourced from Kaggle. It contains $768$ instances and $8$ numerical feature variables:

| Feature | Description |
| :--- | :--- |
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration (2 hours in an oral glucose tolerance test) |
| **BloodPressure** | Diastolic blood pressure ($mm\,Hg$) |
| **SkinThickness** | Triceps skin fold thickness ($mm$) |
| **Insulin** | 2-Hour serum insulin ($mu\,U/ml$) |
| **BMI** | Body Mass Index ($weight\,in\,kg / (height\,in\,m)^2$) |
| **DiabetesPedigreeFunction** | Scores likelihood of diabetes based on family history |
| **Age** | Age in years |
| **Outcome** | Class variable ($0$: Non-Diabetic, $1$: Diabetic) |

---

## 🔍 Exploratory Data Analysis (EDA)
The analysis incorporates visual checks to understand class distribution and variable ranges:
1. **Target Distribution:** Evaluated class balance using `sns.countplot()`.
2. **Outlier Detection:** Leveraged `sns.boxplot()` across all continuous features.
3. **Feature Distributions:** Inspected feature skewness with KDE histogram overlays (`sns.histplot()`).
4. **Multivariate Relationships:** Analyzed feature interactions split by target class using `sns.pairplot()`.

---

## ⚙️ Preprocessing & Model Architecture

### Pipeline Steps:
1. **Feature Scaling:** Since KNN relies on distance metrics (Euclidean distance), features were standardized using `StandardScaler` to ensure uniform feature magnitude.
2. **Train-Test Split:** Partitioned the dataset into **70% Training** and **30% Testing** subsets (`random_state=42`).
3. **Hyperparameter Tuning:** Iterated through $K=1$ to $K=14$ neighbors to track training vs. test accuracy trade-offs.
4. **Final Model:** Selected $K=13$ as the optimal neighbor threshold.

---

## 📈 Model Evaluation & Results

The final KNN model evaluated on the unseen $30\%$ test dataset yields the following metrics:

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.79      0.80      0.80       151
           1       0.61      0.60      0.61        80

    accuracy                           0.73       231
   macro avg       0.70      0.70      0.70       231
weighted avg       0.73      0.73      0.73       231
```
# Dataset Link :
https://www.kaggle.com/datasets/mragpavank/diabetes
