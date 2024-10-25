'''
Hướng dẫn:
Cập nhật đường dẫn: Đảm bảo đường dẫn đến dữ liệu gốc (data_path) là chính xác.
Thay đổi số lượng mẫu nếu cần thiết: Điều chỉnh n=200 nếu bạn muốn số bản ghi khác.
Lưu dữ liệu: Chạy mã này, và tập dữ liệu mẫu sẽ được lưu vào dataset/new_data/new_transactions.csv.
'''
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def generate_random_data(n_samples):
    # Đường dẫn đến dữ liệu gốc
    data_path = 'dataset/raw/creditcard.csv'  # Cập nhật đúng đường dẫn nếu cần
    data = pd.read_csv(data_path)

    # Tạo DataFrame mới với các cột cần thiết
    new_data = pd.DataFrame()

    # Tạo TransactionID mới
    new_data['TransactionID'] = range(1, n_samples + 1)

    # Tạo TransactionDate ngẫu nhiên trong khoảng 1 năm gần đây
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    new_data['TransactionDate'] = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(n_samples)]

    # Lấy ngẫu nhiên các giá trị cho các cột khác
    new_data['Amount'] = np.random.choice(data['Amount'], n_samples)
    new_data['MerchantID'] = np.random.choice(data['MerchantID'], n_samples)
    new_data['TransactionType'] = np.random.choice(data['TransactionType'], n_samples)
    new_data['Location'] = np.random.choice(data['Location'], n_samples)

    # Thêm cột IsFraud nhưng để trống (NaN)
    new_data['IsFraud'] = np.nan

    # Sắp xếp theo TransactionDate
    new_data = new_data.sort_values('TransactionDate')

    # Reset index để đảm bảo TransactionID theo thứ tự tăng dần
    new_data = new_data.reset_index(drop=True)
    new_data['TransactionID'] = new_data.index + 1

    # Lưu tập dữ liệu mẫu vào thư mục new_data
    sample_data_path = 'dataset/new_data/new_transactions.csv'
    new_data.to_csv(sample_data_path, index=False)

    print(f"New random data saved to {sample_data_path}")

# Chạy hàm để tạo dữ liệu mới
if len(sys.argv) > 1:
    n_samples = int(sys.argv[1])
    generate_random_data(n_samples)
else:
    print("Please provide the number of samples as a command line argument.")