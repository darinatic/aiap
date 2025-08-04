from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .utils import *

def get_rf_param_grid():
    """
    Get Random Forest hyperparameter grid
    """
    return {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

def train_random_forest(X_train, y_train, param_grid=None):
    """
    Train Random Forest with grid search
    """
    if param_grid is None:
        param_grid = get_rf_param_grid()
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def run_random_forest_pipeline(df_clean, test_size=0.2):
    """
    Complete Random Forest pipeline using utils
    """
    print("=== RANDOM FOREST PIPELINE ===")
    
    # Prepare data
    X, y, selected_features = prepare_data(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    
    # Train model
    model, best_params = train_random_forest(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, "Random Forest")
    
    # Plot results
    plot_results(model, X_test, y_test, selected_features, "Random Forest")
    
    # Save model
    save_model(model, 'models/random_forest.pkl', {
        'selected_features': selected_features,
        'best_params': best_params
    })
    
    # Print summary
    print_model_summary("Random Forest", results, selected_features, best_params)
    
    return model, results

if __name__ == "__main__":
    from data_preprocessing import load_and_clean_data
    
    df_clean, _ = load_and_clean_data()
    model, results = run_random_forest_pipeline(df_clean)
    print("Random Forest pipeline completed!")