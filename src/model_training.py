# Remove duplicate imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("c:\\Data Science Project\\credit-card-fraud-detection")
from src.data_preprocessing import load_and_explore_data, clean_dataset, balance_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import joblib
import os
from src.feature_engineering import engineer_all_features

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n=== Detailed Evaluation for {model_name} ===")
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # False Positive Analysis
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    print(f"\nFalse Positive Rate: {fpr:.3f}")
    print(f"Number of False Positives: {fp}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba, model_name)
    
    return {
        'fpr': fpr,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'auc_score': auc(*roc_curve(y_test, y_pred_proba)[:2])
    }

def optimize_hyperparameters(model, X_train, y_train, model_name):
    """
    Perform hyperparameter tuning using GridSearchCV
    """
    # Define parameter grids for each model
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'XGBoost': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'scale_pos_weight': [1, 2, 3]
        }
    }
    
    # Custom scorer that penalizes false positives more heavily
    scoring = {
        'AUC': 'roc_auc',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=5,
        scoring=scoring,
        refit='AUC',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n=== Performing GridSearchCV for {model_name} ===")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(random_state=42, scale_pos_weight=1)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        
        # Cross-validation and model fitting
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-validation ROC-AUC scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        model.fit(X_train, y_train)
        
        # Detailed evaluation
        eval_results = evaluate_model(model, X_test, y_test, name)
        
        results[name] = {
            'cv_scores': cv_scores,
            'model': model,
            'evaluation': eval_results
        }
    
    # Find best performing model based on AUC
    best_model_name = max(results.items(), 
                         key=lambda x: x[1]['evaluation']['auc_score'])[0]
    
    print(f"\n=== Saving {best_model_name} as final model ===")
    best_model = results[best_model_name]['model']
    
    # Create models directory if it doesn't exist
    model_dir = "c:\\Data Science Project\\credit-card-fraud-detection\\models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the best model using both pickle and joblib
    model_path_pkl = os.path.join(model_dir, "final_model.pkl")
    model_path_joblib = os.path.join(model_dir, "final_model.joblib")
    
    # Save using pickle
    with open(model_path_pkl, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save using joblib (more efficient for large numpy arrays)
    joblib.dump(best_model, model_path_joblib)
    
    print(f"Model saved as:\n - {model_path_pkl}\n - {model_path_joblib}")
    
    return results

def prepare_data():
    df = load_and_explore_data()
    df_clean = clean_dataset(df)
    df_featured = engineer_all_features(df_clean)
    return balance_dataset(df_featured)

if __name__ == "__main__":
    # Use the prepare_data function
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)