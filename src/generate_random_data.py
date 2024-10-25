'''
Hướng dẫn:
Cập nhật đường dẫn: Đảm bảo đường dẫn đến dữ liệu gốc (data_path) là chính xác.
Thay đổi số lượng mẫu nếu cần thiết: Điều chỉnh n=200 nếu bạn muốn số bản ghi khác.
Lưu dữ liệu: Chạy mã này, và tập dữ liệu mẫu sẽ được lưu vào dataset/new_data/new_transactions.csv.
'''

import pandas as pd

# Đường dẫn đến dữ liệu gốc
data_path = 'dataset/raw/creditcard.csv'  # Cập nhật đúng đường dẫn nếu cần
data = pd.read_csv(data_path)

# Lấy mẫu ngẫu nhiên 200 bản ghi từ dữ liệu gốc
sample_data = data.sample(n=200, random_state=42)  # Bạn có thể thay đổi số lượng mẫu nếu muốn

# Lưu tập dữ liệu mẫu vào thư mục new_data
sample_data_path = 'dataset/new_data/new_transactions.csv'
sample_data.to_csv(sample_data_path, index=False)

print(f"Sample data saved to {sample_data_path}")
