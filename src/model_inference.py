'''
Giải thích từng phần mã:
Nạp mô hình và bộ tiền xử lý: Tải mô hình Random Forest hoặc Logistic Regression và bộ tiền xử lý để chuẩn bị cho quá trình suy diễn.

Nạp dữ liệu mới:

Đọc dữ liệu mới từ dataset/new_data/new_transactions.csv.
Áp dụng các bước xử lý thời gian và loại bỏ các cột không cần thiết (TransactionID, TransactionDate, MerchantID).
Tiền xử lý dữ liệu mới:

Sử dụng preprocessor đã lưu để áp dụng các bước chuẩn hóa và mã hóa trên dữ liệu mới.
Dự đoán:

Tạo dự đoán bằng cách sử dụng model.predict() cho nhãn dự đoán và model.predict_proba() cho xác suất gian lận.
Gán các cột IsFraudPrediction và FraudProbability cho dữ liệu mới.
Lưu kết quả:

Lưu kết quả dự đoán vào tệp CSV predictions.csv trong thư mục dataset/new_data/.
'''
import pandas as pd
import joblib

# Load the pre-trained models and preprocessor
logistic_model = joblib.load('models/logistic_model.pkl')
random_forest_model = joblib.load('models/random_forest_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Load new data
new_data_path = 'dataset/new_data/new_transactions.csv'
new_data = pd.read_csv(new_data_path)

# Preprocess new data
new_data['TransactionDate'] = pd.to_datetime(new_data['TransactionDate'])
new_data['Day'] = new_data['TransactionDate'].dt.day
new_data['Month'] = new_data['TransactionDate'].dt.month
new_data['Hour'] = new_data['TransactionDate'].dt.hour
new_data = new_data.drop(columns=['TransactionID', 'TransactionDate', 'MerchantID'])

# Transform new data using the saved preprocessor
X_new = preprocessor.transform(new_data)

def make_predictions(model, model_name):
    # Perform inference
    predictions = model.predict(X_new)
    prediction_probabilities = model.predict_proba(X_new)[:, 1]  # Probability of being fraud

    # Prepare results
    results = new_data.copy()
    results['IsFraudPrediction'] = predictions
    results['FraudProbability'] = prediction_probabilities

    # Remove unwanted columns
    columns_to_remove = ['IsFraud', 'Day', 'Month', 'Hour']
    results = results.drop(columns=columns_to_remove, errors='ignore')

    # Save results to a CSV file
    output_path = f'dataset/new_data/predictions_{model_name}.csv'
    results.to_csv(output_path, index=False)
    print(f"Predictions for {model_name} saved to {output_path}")

# Make predictions for both models
make_predictions(logistic_model, "logistic_regression")
make_predictions(random_forest_model, "random_forest")