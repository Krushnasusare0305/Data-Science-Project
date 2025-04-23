import os
import sys
import joblib
import numpy as np
import pandas as pd
import traceback  # Add traceback for better error reporting

# Add current directory to path
sys.path.append(os.getcwd())

def load_model(model_path='models/trained_model.pkl'):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model

def preprocess_transaction(transaction, model_features=None):
    """Apply the same preprocessing steps to a new transaction"""
    # Create DataFrame if input is a dictionary
    if isinstance(transaction, dict):
        transaction = pd.DataFrame([transaction])
    
    # Add hour feature
    transaction['Hour'] = transaction['Time'] % (24 * 3600) / 3600
    
    # Log transform amount
    transaction['LogAmount'] = np.log1p(transaction['Amount'])
    
    # Drop Time column if it exists
    if 'Time' in transaction.columns:
        transaction = transaction.drop('Time', axis=1)
    
    # Ensure feature order matches the model's expected features
    if model_features is not None:
        # Create a DataFrame with all model features, filled with zeros
        transaction_aligned = pd.DataFrame(0, index=transaction.index, columns=model_features)
        
        # Fill in the values for features that exist in the transaction
        for col in transaction.columns:
            if col in model_features:
                transaction_aligned[col] = transaction[col]
        
        return transaction_aligned
    
    return transaction

def get_model_features():
    """Get a list of features used during model training"""
    # Read a small sample to get the feature names
    df = pd.read_csv('data/creditcard.csv', nrows=1)
    
    # Apply the same preprocessing as during training
    df['Hour'] = df['Time'] % (24 * 3600) / 3600
    df['LogAmount'] = np.log1p(df['Amount'])
    df = df.drop(['Time', 'Class'], axis=1)
    
    # Return the list of feature names
    return df.columns.tolist()

def predict_fraud(model, transaction_data, threshold=0.5):
    """Predict whether a transaction is fraudulent"""
    try:
        # Get the features used during training
        model_features = get_model_features()
        
        # Preprocess the transaction data
        preprocessed_data = preprocess_transaction(transaction_data, model_features)
        
        # Make prediction
        prediction = model.predict(preprocessed_data)[0]
        probability = model.predict_proba(preprocessed_data)[0, 1]
        
        result = {
            'prediction': int(prediction),
            'fraud_probability': float(probability),
            'is_fraud': bool(probability >= threshold),
            'transaction_amount': float(transaction_data['Amount'])
        }
        
        return result
    except Exception as e:
        print(f"Error in predict_fraud: {e}")
        traceback.print_exc()
        return None

def create_sample_transactions(n=5):
    """Create sample transactions for testing"""
    # Normal transaction (low probability of fraud)
    normal_transaction = {
        'Time': 80000,
        'V1': 1.0, 'V2': 0.5, 'V3': 1.1, 'V4': 0.8,
        'V5': 0.5, 'V6': 0.2, 'V7': 0.4, 'V8': 0.1,
        'V9': 0.3, 'V10': 0.1, 'V11': 0.5, 'V12': 0.3,
        'V13': 0.2, 'V14': 0.4, 'V15': 0.5, 'V16': 0.1,
        'V17': 0.3, 'V18': 0.2, 'V19': 0.1, 'V20': 0.2,
        'V21': 0.1, 'V22': 0.3, 'V23': 0.1, 'V24': 0.1,
        'V25': 0.1, 'V26': 0.2, 'V27': 0.1, 'V28': 0.1,
        'Amount': 25.0
    }
    
    # Suspicious transaction (with negative V1, V3 values - often associated with fraud)
    suspicious_transaction = {
        'Time': 40000,
        'V1': -3.0, 'V2': -2.5, 'V3': -4.0, 'V4': -2.5,
        'V5': -1.5, 'V6': -0.5, 'V7': -1.0, 'V8': -0.5,
        'V9': -2.0, 'V10': -1.0, 'V11': -1.5, 'V12': -0.5,
        'V13': -1.0, 'V14': -2.0, 'V15': -1.5, 'V16': -0.5,
        'V17': -1.0, 'V18': -0.5, 'V19': -1.0, 'V20': -0.5,
        'V21': -1.5, 'V22': -1.0, 'V23': -0.5, 'V24': -0.5,
        'V25': -1.0, 'V26': -0.5, 'V27': -1.0, 'V28': -0.5,
        'Amount': 3500.0  # Large amount
    }
    
    # Create a few random transactions
    transactions = [normal_transaction, suspicious_transaction]
    
    # Add a few more random transactions
    for _ in range(n-2):
        random_transaction = {
            'Time': np.random.randint(0, 172800),
            'Amount': np.random.uniform(1, 1000)
        }
        
        # Add random values for V1-V28
        for i in range(1, 29):
            random_transaction[f'V{i}'] = np.random.uniform(-2, 2)
            
        transactions.append(random_transaction)
    
    return transactions

def main():
    try:
        # Load the trained model
        model = load_model()
        
        # Create sample transactions
        transactions = create_sample_transactions()
        
        # Make predictions
        print("\nMaking predictions on sample transactions:")
        print("-" * 60)
        
        for i, transaction in enumerate(transactions):
            print(f"Processing transaction {i+1}...")
            
            result = predict_fraud(model, transaction)
            
            if result:
                print(f"Transaction {i+1}:")
                print(f"  Amount: ${result['transaction_amount']:.2f}")
                print(f"  Fraud Probability: {result['fraud_probability']:.6f}")
                print(f"  Prediction: {'FRAUD' if result['is_fraud'] else 'NORMAL'}")
                print("-" * 60)
            else:
                print(f"Failed to process transaction {i+1}")
                print("-" * 60)
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 