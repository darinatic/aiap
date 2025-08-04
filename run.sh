#!/bin/bash

echo "Starting AIAP Assessment 21 ML Pipeline..."

echo "Creating output directories..."
mkdir -p models

echo "Running data preprocessing and model training..."
python -c "
from src.data_preprocessing import load_and_clean_data
from src.random_forest_model import run_random_forest_pipeline
from src.svm_model import run_svm_pipeline
from src.xgboost_model import run_xgboost_pipeline

print('Loading and cleaning data...')
df_clean, preprocessor = load_and_clean_data()

print('Running all model pipelines...')
rf_model, rf_results = run_random_forest_pipeline(df_clean)
svm_model, svm_results = run_svm_pipeline(df_clean)
xgb_model, xgb_results = run_xgboost_pipeline(df_clean)

print('\\n=== FINAL RESULTS SUMMARY ===')
print(f'Random Forest - Accuracy: {rf_results[\"accuracy\"]:.4f}, F1-weighted: {rf_results[\"f1_weighted\"]:.4f}')
print(f'SVM - Accuracy: {svm_results[\"accuracy\"]:.4f}, F1-weighted: {svm_results[\"f1_weighted\"]:.4f}')
print(f'XGBoost - Accuracy: {xgb_results[\"accuracy\"]:.4f}, F1-weighted: {xgb_results[\"f1_weighted\"]:.4f}')
print('Pipeline completed successfully!')
"

echo "ML Pipeline execution completed!"