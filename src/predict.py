import pandas as pd
import pickle
from src.data_preprocessing import clean_dataset
from src.feature_engineering import engineer_all_features

def load_model(model_path='models/final_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_fraud(transaction_data, model):
    """
    Make fraud predictions on new transaction data
    """
    # Preprocess the data
    df_clean = clean_dataset(transaction_data)
    df_featured = engineer_all_features(df_clean)
    
    # Make predictions
    predictions = model.predict(df_featured.drop(['Timestamp'], axis=1))
    probabilities = model.predict_proba(df_featured.drop(['Timestamp'], axis=1))[:, 1]
    
    # Add predictions to dataframe
    df_featured['fraud_prediction'] = predictions
    df_featured['fraud_probability'] = probabilities
    
    return df_featured

if __name__ == "__main__":
    # Example usage
    model = load_model()
    
    # Load new transactions (example)
    new_transactions = pd.read_csv('data/new_transactions.csv')
    
    # Make predictions
    results = predict_fraud(new_transactions, model)
    print("\nPrediction Results:")
    print(results[['Timestamp', 'Amount', 'fraud_prediction', 'fraud_probability']].head())