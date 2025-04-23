import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

def load_model(model_path='models/final_model.pkl'):
    print(f"Attempting to load model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    if hasattr(model, 'shape'):
        print(f"Model shape: {model.shape}")
    return model

def main():
    # Load the model
    print("Loading the model...")
    try:
        model = load_model()
        print("Model loaded successfully!")
        print("Model details:", model)
        
        # If we got this far, let's try to train a simple model to make predictions
        print("\nTraining a simple logistic regression model...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Load a small sample of data
        sample_data = pd.read_csv('data/creditcard.csv', nrows=10000)
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        # Train a simple model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Create a sample transaction
        print("\nCreating a sample transaction...")
        sample_transaction = {
            'Time': 0,
            'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
            'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
            'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
            'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
            'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
            'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
            'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
            'Amount': 149.62
        }
        
        # Convert to DataFrame
        sample_df = pd.DataFrame([sample_transaction])
        
        # Make prediction with our new model
        print("\nMaking prediction...")
        prediction = clf.predict(sample_df)
        probability = clf.predict_proba(sample_df)[:, 1]
        
        print(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
        print(f"Probability of fraud: {probability[0]:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 