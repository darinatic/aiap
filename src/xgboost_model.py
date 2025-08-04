from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from .utils import *

def get_xgb_param_grid():
    return {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

def train_xgboost(X_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = get_xgb_param_grid()
    
    print("Training XGBoost...")
    
    # Encode target labels for XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_encoded)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, label_encoder

def run_xgboost_pipeline(df_clean, test_size=0.2):
    print("=== XGBOOST PIPELINE ===")
    
    X, y, selected_features = prepare_data(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    model, best_params, label_encoder = train_xgboost(X_train, y_train)
    
    # Encode test labels
    y_test_encoded = label_encoder.transform(y_test)
    
    results = evaluate_model_encoded(model, X_test, y_test, y_test_encoded, label_encoder, "XGBoost")
    plot_results(model, X_test, y_test, selected_features, "XGBoost")
    
    save_model(model, 'models/xgboost.pkl', {
        'selected_features': selected_features,
        'best_params': best_params,
        'label_encoder': label_encoder
    })
    
    print_model_summary("XGBoost", results, selected_features, best_params)
    return model, results

def evaluate_model_encoded(model, X_test, y_test, y_test_encoded, label_encoder, model_name="Model"):
    print(f"\nEvaluating {model_name}...")
    
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
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

if __name__ == "__main__":
    from data_preprocessing import load_and_clean_data
    
    df_clean, _ = load_and_clean_data()
    model, results = run_xgboost_pipeline(df_clean)
    print("XGBoost pipeline completed!")