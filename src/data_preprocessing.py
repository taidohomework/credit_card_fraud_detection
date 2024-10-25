'''
Hướng dẫn:
Chạy tệp data_preprocessing.py sau khi đã cập nhật đường dẫn của tệp dữ liệu ban đầu.
Kết quả tiền xử lý sẽ được lưu vào dataset/processed/ dưới dạng các tệp X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl.
Tệp preprocessor.pkl sẽ chứa đối tượng tiền xử lý (OneHotEncoder và StandardScaler) và được lưu trong thư mục models/ để dùng lại khi bạn cần áp dụng mô hình trên dữ liệu mới.
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv('dataset/raw/creditcard.csv')  # Update this path as needed

# Convert TransactionDate to datetime
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

# Extract useful datetime features
data['Day'] = data['TransactionDate'].dt.day
data['Month'] = data['TransactionDate'].dt.month
data['Hour'] = data['TransactionDate'].dt.hour
data = data.drop(columns=['TransactionID', 'TransactionDate', 'MerchantID'])  # Drop unnecessary columns

# Define target and features
X = data.drop(columns=['IsFraud'])
y = data['IsFraud']

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Amount', 'Day', 'Month', 'Hour']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['TransactionType', 'Location'])
    ])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Save the processed data and the preprocessor for later use in training and inference
joblib.dump(X_train, 'dataset/processed/X_train.pkl')
joblib.dump(X_test, 'dataset/processed/X_test.pkl')
joblib.dump(y_train, 'dataset/processed/y_train.pkl')
joblib.dump(y_test, 'dataset/processed/y_test.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
