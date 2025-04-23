import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add current directory to path
sys.path.append(os.getcwd())

def load_data(file_path='data/creditcard.csv', sample_size=None):
    """Load the credit card fraud dataset"""
    print(f"Loading data from {file_path}...")
    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts(normalize=True)}")
    return df

def preprocess_data(df):
    """Apply basic preprocessing to the data"""
    print("\nPreprocessing data...")
    
    # Add a time-based feature
    df['Hour'] = df['Time'] % (24 * 3600) / 3600
    
    # Log transform the Amount feature
    df['LogAmount'] = np.log1p(df['Amount'])
    
    # Drop the Time column
    df = df.drop('Time', axis=1)
    
    return df

def train_model(X_train, y_train):
    """Train a logistic regression model"""
    print("\nTraining logistic regression model...")
    
    # Initialize the model with appropriate parameters for imbalanced data
    model = LogisticRegression(
        C=0.1,                # Regularization strength
        class_weight='balanced', # Handle class imbalance
        max_iter=1000,        # Increase max iterations
        random_state=42       # For reproducibility
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    # Calculate false positive rate
    fpr = fp / (fp + tn)
    print(f"\nFalse Positive Rate: {fpr:.6f}")
    
    # Calculate ROC AUC score
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {auc:.6f}")
    
    return {
        'confusion_matrix': (tn, fp, fn, tp),
        'false_positive_rate': fpr,
        'auc': auc
    }

def save_model(model, file_path='models/trained_model.pkl'):
    """Save the trained model to a file"""
    print(f"\nSaving model to {file_path}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, file_path)
    print(f"Model saved successfully to {file_path}")

def main():
    try:
        # Load data (using a smaller sample for faster training)
        df = load_data(sample_size=50000)  # Use a smaller sample for testing
        
        # Preprocess data
        df = preprocess_data(df)
        
        # Split features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        eval_metrics = evaluate_model(model, X_test, y_test)
        
        # Save the model
        save_model(model)
        
        print("\nModel training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 