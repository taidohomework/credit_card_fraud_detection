import os
import pandas as pd

def show_fraud_statistics():
    models = {
        "Logistic Regression": "predictions_logistic_regression.csv",
        "Random Forest": "predictions_random_forest.csv"
    }
    
    for model_name, file_name in models.items():
        file_path = f"dataset/new_data/{file_name}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            total_transactions = len(df)
            fraud_predictions = df['IsFraudPrediction'].sum()
            fraud_percentage = (fraud_predictions / total_transactions) * 100
            
            print(f"\n{model_name} Model Statistics:")
            print(f"Total Transactions: {total_transactions}")
            print(f"Predicted Fraudulent Transactions: {fraud_predictions}")
            print(f"Percentage of Fraudulent Transactions: {fraud_percentage:.2f}%")
            
            # Hiển thị top 5 giao dịch có xác suất gian lận cao nhất
            top_5_fraud = df.nlargest(5, 'FraudProbability')
            print("\nTop 5 Transactions with Highest Fraud Probability:")
            print(top_5_fraud[['Amount', 'TransactionType', 'Location', 'FraudProbability']])
        else:
            print(f"\nFile not found: {file_path}")

show_fraud_statistics()