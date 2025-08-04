from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.utils import *

def get_svm_param_grid():
    return {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf', 'poly']
    }

def train_svm(X_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = get_svm_param_grid()
    
    print("Training SVM...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, scaler

def evaluate_model_scaled(model, X_test_scaled, y_test, model_name="Model"):
    print(f"\nEvaluating {model_name}...")
    
    y_pred = model.predict(X_test_scaled)
    
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

def run_svm_pipeline(df_clean, test_size=0.2):
    print("=== SVM PIPELINE ===")
    
    X, y, selected_features = prepare_data(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    model, best_params, scaler = train_svm(X_train, y_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    results = evaluate_model_scaled(model, X_test_scaled, y_test, "SVM")
    # plot_results(model, X_test_scaled, y_test, selected_features, "SVM")
    
    save_model(model, 'models/svm.pkl', {
        'selected_features': selected_features,
        'best_params': best_params,
        'scaler': scaler
    })
    
    print_model_summary("SVM", results, selected_features, best_params)
    return model, results

if __name__ == "__main__":
    from src.data_preprocessing import load_and_clean_data
    
    df_clean, _ = load_and_clean_data()
    model, results = run_svm_pipeline(df_clean)
    print("SVM pipeline completed!")