import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

def select_features_from_eda(df):
    """
    Select features based on EDA insights
    """
    selected_features = [
        'MetalOxideSensor_Unit4',     # Highest correlation (0.308)
        'CO2_ElectroChemicalSensor',  # Second highest (0.268)
        'CO_GasSensor',               # Strong negative (-0.229)
        'CO2_InfraredSensor',         # Moderate negative (-0.189)
        'Temperature',                # Moderate negative (-0.162)
        'Time of Day'                 # Weak but might help (-0.092)
    ]
    
    print(f"Selected features based on EDA: {selected_features}")
    return selected_features

def prepare_data(df_clean, selected_features=None, target_col='Activity Level'):
    """
    Prepare features and target for ML
    """
    if selected_features is None:
        selected_features = select_features_from_eda(df_clean)
    
    X = df_clean[selected_features]
    y = df_clean[target_col]
    
    print(f"Data prepared: {len(selected_features)} features, {len(df_clean)} samples")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, selected_features

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance and return metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    
    # Classification report
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
    """
    Plot confusion matrix
    """
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

def plot_feature_importance(feature_names, importances, model_name="Model", ax=None):
    """
    Plot feature importance
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    ax.barh(range(len(importance_df)), importance_df['importance'])
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'{model_name} - Feature Importance')
    ax.invert_yaxis()
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return importance_df

def plot_results(model, X_test, y_test, feature_names, model_name="Model"):
    """
    Plot both confusion matrix and feature importance
    """
    y_pred = model.predict(X_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name, ax=ax1)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(feature_names, model.feature_importances_, model_name, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{model_name} - No Feature Importance')
    
    plt.tight_layout()
    plt.show()

def save_model(model, filepath, additional_data=None):
    """
    Save model with optional additional data
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {'model': model}
    if additional_data:
        model_data.update(additional_data)
    
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load saved model
    """
    model_data = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_data

def print_model_summary(model_name, results, selected_features, best_params=None):
    """
    Print model summary
    """
    print(f"\n=== {model_name.upper()} SUMMARY ===")
    print(f"Selected Features: {selected_features}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
    if best_params:
        print(f"Best Parameters: {best_params}")
    print("=" * 50)