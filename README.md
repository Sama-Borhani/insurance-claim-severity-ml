# Gradient Boosted Insurance Severity Model

This project models insurance claim severity using gradient boosting techniques. Based on the Allstate Claims Severity dataset, the project demonstrates how advanced models like LightGBM can significantly outperform traditional linear models for predicting continuous target variables.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Pipeline](#modeling-pipeline)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Interpretability (SHAP)](#interpretability-shap)
- [Contributors](#contributors)
- [License](#license)

---

##  Project Overview

- Dataset: [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity)
- Problem: Predict insurance claim severity (regression)
- Goal: Reduce Mean Absolute Error (MAE) using machine learning techniques.
- Result: LightGBM reduced MAE by **9.4%** compared to Linear Regression.

---

##  Features

- Data optimization to reduce memory usage
- Exploratory Data Analysis with visualizations
- Feature encoding for 116 categorical variables
- Linear Regression baseline model
- LightGBM regressor as the main model
- SHAP analysis for interpretability
- Model evaluation with MAE/RMSE metrics

---

##  Installation

Make sure to install the required packages:
If running on Google Colab:
!pip install -q kaggle
!pip install -q lightgbm shap

---

##  Usage
1. Download the Dataset
kaggle competitions download -c allstate-claims-severity
unzip allstate-claims-severity.zip -d allstate-claims-severity

2. Load and Preprocess Data
Optimize memory
One-hot encode categorical variables
Train-test split

3. Run the Baseline and LightGBM Models
# Train linear model
LinearRegression().fit(X_train, y_train)

# Train LightGBM
lgb.LGBMRegressor(...).fit(X_train, y_train)

---

##  Modeling Pipeline

1. Data ingestion and type optimization
2. EDA: histograms, correlations, and distributions
3. Encoding: one-hot for categorical variables
4. Baseline Model: Ordinary Least Squares (Linear Regression)
5. LightGBM: gradient boosting regressor
6. Performance Evaluation: MAE and RMSE

---

##  Evaluation
| Metric      | Linear Regression | LightGBM  |
| ----------- | ----------------- | --------- |
| MAE (Train) | \$1302.63         | \$1033.49 |
| MAE (Val)   | \$1292.53         | \$1171.03 |
| RMSE (Val)  | \$2010.50         | \$1859.39 |

MAE Improvement: 9.4%
RMSE Improvement: 7.5%
---

##  Feature Importance
Top 5 Features by LightGBM:
1. cont7
2. cont14
3. cont2
4. cont12
5. cont11
Continuous variables dominate, indicating geography-based premium refinements might be valuable.

---

##  Feature Importance
SHAP values were calculated to explain LightGBM predictions:
Visualized the top 20 feature impacts
Enabled model transparency and trust
Example:
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
