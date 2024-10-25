'''
Giải thích chi tiết từng phần trong mã:
Nạp dữ liệu kiểm tra: Sử dụng X_test và y_test từ thư mục dataset/processed/ để đánh giá mô hình trên tập kiểm tra.

Nạp mô hình đã huấn luyện: Tải các mô hình Logistic Regression và Random Forest từ thư mục models/.

Hàm evaluate_model:

Dự đoán các nhãn (y_pred) và xác suất (y_proba) để tính các chỉ số hiệu suất.
Tính toán các chỉ số đánh giá (accuracy, ROC AUC, classification report).
Hiển thị ma trận nhầm lẫn bằng cách sử dụng seaborn để trực quan hóa số lượng dự đoán đúng và sai của từng lớp.
Gọi hàm evaluate_model: Chạy hàm này cho cả hai mô hình và hiển thị kết quả.
'''

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed test data
X_test = joblib.load('dataset/processed/X_test.pkl')
y_test = joblib.load('dataset/processed/y_test.pkl')

# Load the trained models
logistic_model = joblib.load('models/logistic_model.pkl')
random_forest_model = joblib.load('models/random_forest_model.pkl')

# Define a function to evaluate and print metrics
def evaluate_model(model, X_test, y_test, model_name):
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print("Classification Report:\n", class_report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Evaluate Logistic Regression model
evaluate_model(logistic_model, X_test, y_test, "Logistic Regression")

# Evaluate Random Forest model
evaluate_model(random_forest_model, X_test, y_test, "Random Forest")
