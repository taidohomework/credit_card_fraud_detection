# Credit Card Fraud Detection

## Mô tả dự án
Dự án này nhằm xây dựng một hệ thống phát hiện gian lận thẻ tín dụng bằng cách sử dụng các mô hình học máy (Machine Learning) như Logistic Regression và Random Forest. Hệ thống bao gồm các chức năng chính: tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá hiệu suất mô hình và dự đoán trên dữ liệu mới.

Nguồn dữ liệu được sử dụng cho dự án có sẵn trên Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/bhadramohit/credit-card-fraud-detection).

## Hướng dẫn cài đặt

### Yêu cầu
- Python >= 3.7
- Các thư viện được liệt kê trong `requirements.txt`

### Cài đặt
1. Tạo môi trường ảo:
   ```bash
   python -m venv venv
   ```
2. Kích hoạt môi trường ảo:
   - **Windows**: `venv\Scripts\activate`
   - **MacOS/Linux**: `source venv/bin/activate`

3. Cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```

### Chạy dự án
```bash
python main.py
```

### Các chức năng chính
1. **Preprocess Data**: Tiền xử lý dữ liệu từ tập gốc, bao gồm chuẩn hóa và mã hóa biến phân loại. Dữ liệu sau khi tiền xử lý sẽ được lưu vào thư mục `dataset/processed/`.

2. **Train Model**: Huấn luyện mô hình Logistic Regression và Random Forest trên dữ liệu đã tiền xử lý. Các mô hình sẽ được lưu trong thư mục `models/`.

3. **Evaluate Model**: Đánh giá hiệu suất của mô hình trên tập kiểm tra, bao gồm các chỉ số: Accuracy, Precision, Recall, F1-Score, ROC AUC, và Confusion Matrix, Outlier Detection, ROC Curve.

4. **Generate Random Data for Inference**: 
   - Sử dụng `generate_random_data.py` trong `src/` để tạo dữ liệu mẫu ngẫu nhiên từ dữ liệu gốc.
   - Tệp dữ liệu mẫu (`new_transactions.csv`) sẽ được lưu trong thư mục `dataset/new_data/` để dùng cho suy luận.

5. **Make Inference on New Data**: Dự đoán trên dữ liệu mới trong thư mục `dataset/new_data/`. Kết quả dự đoán sẽ được lưu vào `predictions.csv` trong cùng thư mục.

6. **Show Fraud Statistics**: Hiển thị thông tin thống kê và trực quan hóa về các mẫu gian lận.

7. **Exit**: Thoát chương trình.

## Tham khảo
- Nguồn dữ liệu: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/bhadramohit/credit-card-fraud-detection)
- Các mô hình và thuật toán: Logistic Regression, Random Forest
- Các kỹ thuật xử lý dữ liệu mất cân bằng: `class_weight='balanced'`, phân tích và điều chỉnh mô hình cho dữ liệu bất cân đối.

## Tác giả
Dự án được phát triển nhằm mục đích học tập và nghiên cứu các kỹ thuật phát hiện gian lận sử dụng học máy.