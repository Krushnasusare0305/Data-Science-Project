"""
Credit Card Fraud Detection - Problem Definition

Objective:
- Develop a binary classification model to identify fraudulent credit card transactions
- Optimize for minimal false positives while maintaining high fraud detection accuracy

Problem Characteristics:
1. Type: Binary Classification (Fraud vs. Non-Fraud)
2. Target Variable: Transaction Class (0: Normal, 1: Fraudulent)

Key Requirements:
1. Performance Metrics:
    - Primary: False Positive Rate (FPR)
    - Secondary: 
        - Precision
        - Recall
        - F1-Score
        - ROC-AUC

2. Business Constraints:
    - Minimize false fraud alerts (false positives)
    - Real-time prediction capability
    - Handle class imbalance
    - Process sensitive financial data securely

3. Model Deliverables:
    - Trained classification model
    - Feature importance analysis
    - Performance metrics report
    - Model interpretation guidelines
"""

# Define key performance indicators
target_metrics = {
    'false_positive_rate': 0.01,  # Maximum acceptable FPR
    'recall_minimum': 0.85,       # Minimum recall for fraud detection
    'auc_threshold': 0.95,        # Minimum ROC-AUC score
}

# Define class labels
CLASS_LABELS = {
    0: 'Normal Transaction',
    1: 'Fraudulent Transaction'
}