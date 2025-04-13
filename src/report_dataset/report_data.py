import pandas as pd

# Đọc dữ liệu sau khi loại bỏ trùng lặp
data_train = pd.read_csv('src/data_processing/feature/data_train_scaled.csv')

# Tạo bảng thống kê cơ bản
report = data_train.describe().T  # Tạo bảng thống kê và xoay
report["median"] = data_train.median()  # Thêm cột trung vị
report["dtype"] = data_train.dtypes  # Kiểu dữ liệu
report["nan"] = data_train.isnull().sum()  # Số lượng NaN

# Reset index để đưa tên đặc trưng thành một cột
report = report.reset_index()
report = report.rename(columns={"index": "feature_name"})

# In báo cáo
print(report)

# Lưu file CSV, có cột 'feature_name'
report.to_csv('src/report_dataset/report_data_train_scaled.csv', index=False)
