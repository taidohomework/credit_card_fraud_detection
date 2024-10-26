'''
Giải thích chi tiết từng phần trong mã:
Nạp dữ liệu kiểm tra: Sử dụng X_test và y_test từ thư mục dataset/processed/ để đánh giá mô hình trên tập kiểm tra.

Nạp mô hình đã huấn luyện: Tải các mô hình Logistic Regression và Random Forest từ thư mục models/.

Hàm evaluate_model:

Dự đoán các nhãn (y_pred) và xác suất (y_proba) để tính các chỉ số hiệu suất.
Tính toán các chỉ số đánh giá (accuracy, ROC AUC, classification report).
Hiển thị ma trận nhầm lẫn bằng cách sử dụng seaborn để trực quan hóa số lượng dự đoán đúng và sai của từng lớp.
Gọi hàm evaluate_model: Chạy hàm này cho cả hai mô hình và hiển thị kết quả.

1. Thêm hàm plot_outlier_detection:
Hàm này tạo một biểu đồ phân tán cho mỗi mô hình, hiển thị xác suất dự đoán của từng mẫu.
Màu sắc của các điểm phản ánh lớp thực tế (fraud hoặc không fraud).
Các mẫu bị phân loại sai được đánh dấu bằng vòng tròn đen.
2. Thêm hàm plot_roc_curve:
Hàm này tạo một biểu đồ ROC cho cả hai mô hình trên cùng một đồ thị.
Nó cho phép so sánh trực quan hiệu suất của hai mô hình.
3. Sửa đổi hàm evaluate_model:
Thêm lời gọi đến plot_outlier_detection để hiển thị biểu đồ outlier detection cho mỗi mô hình.
Trả về y_proba để sử dụng trong việc vẽ đường cong ROC.
4. Ở cuối file:
Gọi evaluate_model cho cả hai mô hình và lưu kết quả y_proba.
Gọi plot_roc_curve để hiển thị đường cong ROC cho cả hai mô hình.
'''
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    # Outlier detection
    plot_outlier_detection(model, X_test, y_test, model_name)

    return y_proba

def plot_outlier_detection(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_proba, c=y_test, cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Actual Class')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability of Fraud')
    plt.title(f'{model_name} - Outlier Detection')
    
    # Highlight misclassifications
    misclassified = y_test != y_pred
    plt.scatter(np.where(misclassified)[0], y_proba[misclassified], 
                facecolors='none', edgecolors='black', s=100, label='Misclassified')

    plt.legend()
    plt.show()

def plot_roc_curve(y_test, y_proba_lr, y_proba_rf):
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Evaluate Logistic Regression model
y_proba_lr = evaluate_model(logistic_model, X_test, y_test, "Logistic Regression")

# Evaluate Random Forest model
y_proba_rf = evaluate_model(random_forest_model, X_test, y_test, "Random Forest")

# Plot ROC curve for both models
plot_roc_curve(y_test, y_proba_lr, y_proba_rf)
