'''
Hướng dẫn từng phần:
Nạp dữ liệu: Tải dữ liệu đã qua xử lý từ dataset/processed/ và tách các tập huấn luyện và kiểm tra (X_train, X_test, y_train, y_test).

Khởi tạo mô hình:

Logistic Regression: Thiết lập với random_state=42 và max_iter=1000 để đảm bảo mô hình hội tụ.
Random Forest: Thiết lập với random_state=42 và n_estimators=100.
Huấn luyện mô hình:

Huấn luyện từng mô hình trên dữ liệu huấn luyện.
Sử dụng dữ liệu kiểm tra để dự đoán và đánh giá mô hình.
Đánh giá mô hình:

Tính toán các chỉ số đánh giá như accuracy và ROC AUC.
In classification report để có thêm các chỉ số như precision, recall, và F1-score.
Lưu mô hình:

Lưu cả hai mô hình đã huấn luyện vào thư mục models/ dưới dạng logistic_model.pkl và random_forest_model.pkl.

Tham khảo: https://www.kaggle.com/mlg-ulb/creditcardfraud
'''

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# Load preprocessed data
X_train = joblib.load('dataset/processed/X_train.pkl')
X_test = joblib.load('dataset/processed/X_test.pkl')
y_train = joblib.load('dataset/processed/y_train.pkl')
y_test = joblib.load('dataset/processed/y_test.pkl')

# Initialize models with balanced class weights
logistic_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')

# Train Logistic Regression model
print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate Logistic Regression model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
roc_auc_logistic = roc_auc_score(y_test, y_pred_logistic)
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_logistic}")
print(f"ROC AUC: {roc_auc_logistic}")
print("Classification Report:\n", classification_report(y_test, y_pred_logistic))

# Save the trained logistic model
joblib.dump(logistic_model, 'models/logistic_model.pkl')

# Train Random Forest model
print("\nTraining Random Forest...")
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy_rf}")
print(f"ROC AUC: {roc_auc_rf}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save the trained random forest model
joblib.dump(random_forest_model, 'models/random_forest_model.pkl')
