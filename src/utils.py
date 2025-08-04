import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

def select_features_from_eda(df):
    selected_features = [
        'MetalOxideSensor_Unit4',
        'CO2_ElectroChemicalSensor',
        'CO_GasSensor',
        'MetalOxideSensor_Unit2',
        'CO2_InfraredSensor',
        'Temperature'
    ]
    return selected_features

def prepare_data(df_clean, selected_features=None, target_col='Activity Level'):
    if selected_features is None:
        selected_features = select_features_from_eda(df_clean)
    
    X = df_clean[selected_features]
    y = df_clean[target_col]
    
    print(f"Data prepared: {len(selected_features)} features, {len(df_clean)} samples")
    return X, y, selected_features

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name="Model"):
    print(f"\nEvaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    
    print(f"\nClassification Report:")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': class_report,
        'predictions': y_pred
    }

def plot_confusion_matrix(y_test, y_pred, model_name="Model", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = sorted(y_test.unique())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=unique_labels, 
               yticklabels=unique_labels, ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return cm

def plot_results(model, X_test, y_test, feature_names, model_name="Model"):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(y_test, y_pred, model_name)
    plt.show()

def save_model(model, filepath, additional_data=None):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {'model': model}
    if additional_data:
        model_data.update(additional_data)
    
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    model_data = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_data

def print_model_summary(model_name, results, selected_features, best_params=None):
    print(f"\n=== {model_name.upper()} SUMMARY ===")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
    if best_params:
        print(f"Best Parameters: {best_params}")
    print("=" * 50)