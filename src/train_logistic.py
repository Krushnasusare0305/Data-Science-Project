import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
sys.path.append("c:\\Data Science Project\\credit-card-fraud-detection")

from src.data_preprocessing import load_and_explore_data, clean_dataset, balance_dataset
from src.feature_engineering import engineer_all_features

def train_logistic_model():
    # Load and prepare data
    df = load_and_explore_data()
    df_clean = clean_dataset(df)
    df_featured = engineer_all_features(df_clean)
    X_train, X_test, y_train, y_test = balance_dataset(df_featured)
    
    # Train logistic regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='saga', max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    model_dir = "c:\\Data Science Project\\credit-card-fraud-detection\\models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "final_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_logistic_model()
    
    # Print model accuracy
    print(f"\nModel accuracy on test set: {model.score(X_test, y_test):.3f}")