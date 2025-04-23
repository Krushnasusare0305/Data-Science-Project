import joblib
import pandas as pd
import numpy as np

def load_model(model_path='c:\\Data Science Project\\credit-card-fraud-detection\\models\\fraud_detection_model.joblib'):
    """
    Load the trained fraud detection model
    """
    return joblib.load(model_path)

def prepare_transaction_data(transaction_data):
    """
    Prepare new transaction data for prediction
    """
    # Convert single transaction to DataFrame if needed
    if not isinstance(transaction_data, pd.DataFrame):
        transaction_data = pd.DataFrame([transaction_data])
    
    # Apply the same preprocessing steps
    df_cleaned = clean_dataset(transaction_data)
    df_featured = engineer_features(df_cleaned)
    
    # Remove the Class column if present
    if 'Class' in df_featured.columns:
        df_featured = df_featured.drop('Class', axis=1)
    
    return df_featured

def predict_transaction(transaction_data, threshold=0.5):
    """
    Predict whether a transaction is fraudulent
    """
    model = load_model()
    
    # Prepare the data
    prepared_data = prepare_transaction_data(transaction_data)
    
    # Make prediction
    fraud_probability = model.predict_proba(prepared_data)[:, 1]
    is_fraud = fraud_probability >= threshold
    
    return {
        'is_fraud': bool(is_fraud[0]),
        'fraud_probability': float(fraud_probability[0]),
        'timestamp': pd.Timestamp.now()
    }

if __name__ == "__main__":
    # Example usage
    sample_transaction = {
        'Time': 0,
        'V1': 0.0,
        'V2': 0.0,
        # ... add other features ...
        'Amount': 100.0
    }
    
    try:
        result = predict_transaction(sample_transaction)
        print("\nPrediction Result:")
        print(f"Is Fraudulent: {result['is_fraud']}")
        print(f"Fraud Probability: {result['fraud_probability']:.3f}")
        print(f"Prediction Time: {result['timestamp']}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")


import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Create time-based features from the timestamp
    """
    df_time = df.copy()
    df_time['Hour'] = df_time['Timestamp'].dt.hour
    df_time['DayOfWeek'] = df_time['Timestamp'].dt.dayofweek
    df_time['IsWeekend'] = df_time['DayOfWeek'].isin([5, 6]).astype(int)
    
    return df_time

def create_amount_features(df):
    """
    Create amount-based features
    """
    df_amount = df.copy()
    
    # Transaction statistics per card
    df_amount['AmountRank'] = df_amount.groupby('V1')['Amount'].rank(pct=True)
    df_amount['AmountToMean'] = df_amount['Amount'] / df_amount.groupby('V1')['Amount'].transform('mean')
    
    # Global amount features
    df_amount['LogAmount'] = np.log1p(df_amount['Amount'])
    
    return df_amount

def create_velocity_features(df):
    """
    Create velocity-based features
    """
    df_velocity = df.copy()
    
    # Transaction counts in different time windows
    for window in ['30min', '1h', '3h']:
        df_velocity[f'Transactions_{window}'] = df_velocity.sort_values('Timestamp').groupby('V1').apply(
            lambda x: x.rolling(window, on='Timestamp')['Amount'].count()
        ).reset_index(level=0, drop=True)
    
    return df_velocity

def create_location_features(df):
    """
    Create location-based features using V1-V28
    """
    df_location = df.copy()
    
    # Euclidean distances between consecutive transactions
    for i in range(1, 29, 2):
        if f'V{i}' in df_location.columns and f'V{i+1}' in df_location.columns:
            df_location[f'Dist_V{i}_{i+1}'] = np.sqrt(
                df_location[f'V{i}'].pow(2) + df_location[f'V{i+1}'].pow(2)
            )
    
    return df_location

def engineer_all_features(df):
    """
    Apply all feature engineering steps
    """
    df_featured = df.copy()
    
    df_featured = create_time_features(df_featured)
    df_featured = create_amount_features(df_featured)
    df_featured = create_velocity_features(df_featured)
    df_featured = create_location_features(df_featured)
    
    # Fill any remaining NaN values
    df_featured = df_featured.fillna(0)
    
    return df_featured