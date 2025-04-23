import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append(os.getcwd())

def load_data(file_path='data/creditcard.csv', sample_size=10000):
    """Load data from CSV file"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, nrows=sample_size)
    print(f"Data shape: {df.shape}")
    
    # Show class distribution
    print("\nClass distribution:")
    class_dist = df['Class'].value_counts(normalize=True)
    print(class_dist)
    
    return df

def preprocess_data(df):
    """Apply preprocessing to the data"""
    print("\nPreprocessing data...")
    
    # Add hour feature
    df['Hour'] = df['Time'] % (24 * 3600) / 3600
    
    # Log transform amount
    df['LogAmount'] = np.log1p(df['Amount'])
    
    # Drop the Time column
    df = df.drop('Time', axis=1)
    
    return df

def load_model(model_path='models/trained_model.pkl'):
    """Load the trained model"""
    print(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)
    return model

def plot_roc_curve(y_test, y_pred_prob):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('models/roc_curve.png')
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_test, y_pred_prob):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('models/pr_curve.png')
    plt.show()
    
    return pr_auc

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.show()
    
    return cm

def analyze_feature_importance(model, feature_names):
    """Plot feature importance"""
    if hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_[0]
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefficients)})
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
    else:
        print("Model does not provide feature importance")
        return None
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.show()
    
    return feature_importance

def main():
    try:
        # Load and preprocess data
        df = load_data()
        df_preprocessed = preprocess_data(df)
        
        # Split data
        X = df_preprocessed.drop('Class', axis=1)
        y = df_preprocessed['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Load model
        model = load_model()
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Visualize model performance
        print("\nVisualizing model performance...")
        roc_auc = plot_roc_curve(y_test, y_pred_prob)
        pr_auc = plot_precision_recall_curve(y_test, y_pred_prob)
        cm = plot_confusion_matrix(y_test, y_pred)
        
        # Analyze feature importance
        feature_importance = analyze_feature_importance(model, X.columns)
        if feature_importance is not None:
            print("\nTop 10 most important features:")
            print(feature_importance.head(10))
        
        print("\nVisualization completed successfully!")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 