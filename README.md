# msm-depression-ml
Code for ML-based depression prediction among MSM in Ghana using sociodemographic and psychosocial features. Includes preprocessing, feature selection, model training (Random Forest, XGBoost, etc.), evaluation metrics, and  feature importance analysis.

# Depression Prediction among MSM in Ghana Using ML

This repository contains the code, data preprocessing steps, and modeling pipeline for the manuscript titled **"Predicting Depression among Men Who Have Sex with Men in Ghana Using Machine Learning Algorithms"**, submitted to *PLOS Mental Health*.

## Repository Structure

- `data/`: Contains the (anonymized) dataset or a synthetic sample if data is restricted.
- `notebooks/`: Jupyter or R notebooks for exploratory data analysis and preprocessing.
- `models/`: Scripts for training and evaluating the machine learning classifiers.
- `results/`: Evaluation metrics, plots (ROC curves, SHAP), and model outputs.
- `figures/`: Publication-ready figures and SHAP summary plots.

## Methodology Summary

- **Algorithms Used**: Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost, Gradient Boosting, Decision Tree.
- **Key Features**: Perceived stress, social isolation, stigma, demographic info.
- **Techniques**: One-hot encoding, SMOTE (for class imbalance), cross-validation.
- **Evaluation**: Accuracy, ROC AUC, Precision, Recall, F1-score.
- **Feature Importance**: SHAP values.

## Requirements

```bash
scikit-learn
xgboost
lightgbm
catboost
shap
pandas
numpy
matplotlib
seaborn
