# Running the Credit Card Fraud Detection Project

This guide provides step-by-step instructions for running the credit card fraud detection project. The project implements a machine learning model to detect fraudulent credit card transactions.

## Prerequisites

Make sure you have the following dependencies installed:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
xgboost>=1.4.2
imbalanced-learn>=0.8.0
joblib>=1.0.1
matplotlib>=3.4.2
seaborn>=0.11.1
```

You can install them using pip:

```
pip install -r requirements.txt
```

## Dataset

The project uses the credit card fraud dataset which should be located in the `data/` folder as `creditcard.csv`. This dataset contains anonymized credit card transactions with the following features:
- Features V1-V28: Principal components obtained from the original transaction data
- 'Time': Seconds elapsed between transactions
- 'Amount': Transaction amount
- 'Class': Target variable (1 for fraud, 0 for normal)

## Project Structure

```
credit-card-fraud-detection/
├── data/                # Dataset directory
│   └── creditcard.csv   # Credit card fraud dataset
├── models/              # Saved models
│   └── trained_model.pkl # Our trained model
├── src/                 # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── predict.py
│   └── train_logistic.py
├── notebooks/           # Jupyter notebooks
├── requirements.txt     # Project dependencies
├── train_model.py       # Script to train a model
├── make_prediction.py   # Script to make predictions
└── visualize_model.py   # Script to visualize model performance
```

## Running the Project

### 1. Training a Model

To train a logistic regression model on the credit card data:

```
python train_model.py
```

This will:
- Load a subset of the data
- Preprocess the data
- Train a logistic regression model
- Evaluate the model performance
- Save the model to `models/trained_model.pkl`

### 2. Making Predictions

To make predictions using the trained model:

```
python make_prediction.py
```

This will:
- Load the trained model
- Generate sample transactions (normal and suspicious)
- Make predictions on these transactions
- Display the results

### 3. Visualizing Model Performance

To visualize the model's performance:

```
python visualize_model.py
```

This will:
- Load a subset of the data
- Load the trained model
- Generate and save visualizations:
  - ROC curve
  - Precision-Recall curve
  - Confusion matrix
  - Feature importance

The visualizations will be saved in the `models/` directory.

## Customizing the Project

### Modifying Sample Size

By default, the scripts use a subset of the data for faster processing. You can modify the sample size in the scripts:

```python
# In train_model.py, change:
df = load_data(sample_size=50000)  # Change to a different value or None for all data
```

### Adjusting Model Parameters

You can modify the model parameters in `train_model.py`:

```python
# In train_model.py, adjust these parameters:
model = LogisticRegression(
    C=0.1,                # Regularization strength
    class_weight='balanced', # Handle class imbalance
    max_iter=1000,        # Increase max iterations
    random_state=42       # For reproducibility
)
```

### Creating Your Own Transactions

To test the model with your own transaction data, modify the `create_sample_transactions` function in `make_prediction.py`. 