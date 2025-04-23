import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_explore_data(file_path='data/creditcard.csv'):
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("\n=== First 5 rows of the dataset ===")
    print(df.head())
    
    print("\n=== Dataset Info ===")
    print(df.info())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Descriptive Statistics ===")
    print(df.describe())
    
    # Display class distribution
    print("\n=== Class Distribution ===")
    print(df['Class'].value_counts(normalize=True))
    
    return df

def clean_dataset(df):
    df_clean = df.copy()
    df_clean['Timestamp'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df_clean['Time'], unit='s')
    df_clean.drop('Time', axis=1, inplace=True)
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
    
    df_clean.drop_duplicates(keep='first', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def engineer_features(df):
    df_featured = df.copy()
    
    df_featured['TransactionFrequency'] = df_featured.groupby(
        pd.Grouper(key='Timestamp', freq='h')
    )['Amount'].transform('count')
    
    df_featured['MeanAmount'] = df_featured.groupby('V1')['Amount'].transform('mean')
    df_featured['AmountDeviation'] = abs(df_featured['Amount'] - df_featured['MeanAmount'])
    
    amount_std = df_featured.groupby('V1')['Amount'].transform('std')
    amount_mean = df_featured.groupby('V1')['Amount'].transform('mean')
    df_featured['AmountZScore'] = np.where(
        amount_std == 0,
        0,
        (df_featured['Amount'] - amount_mean) / amount_std
    )
    
    df_featured['TransactionVelocity'] = df_featured.sort_values('Timestamp').groupby('V1').apply(
        lambda x: x.rolling('1h', on='Timestamp')['Amount'].count()
    ).reset_index(level=0, drop=True).fillna(0)
    
    df_featured['LocationDeviation'] = np.sqrt(
        df_featured['V2'].pow(2) + df_featured['V3'].pow(2)
    )
    
    df_featured = df_featured.fillna(0)
    
    return df_featured

def balance_dataset(df, target_column='Class', random_state=42):
    X = df.drop([target_column, 'Timestamp'], axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test