# Satellite Orbital Collision Risk Prediction

## Overview
This project predicts satellite collision risk levels using the Collision Avoidance Challenge dataset.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Place raw Kaggle files in:
   data/raw/

3. Run notebooks in order:
   notebooks/01_eda_preprocessing.ipynb
   notebooks/02_feature_engineering.ipynb
   notebooks/03_modeling_and_comparison.ipynb

## Models
- Logistic Regression
- Random Forest
- XGBoost
- Deep Neural Network

## Best Result
XGBoost achieved the best validation Macro F1 score of 0.8809.

## Repository Link
https://github.com/siddhi07/Satellite-Orbital-Collision-Risk-Prediction
