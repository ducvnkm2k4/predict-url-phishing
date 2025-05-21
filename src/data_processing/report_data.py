import pandas as pd

# Đọc dữ liệu sau khi loại bỏ trùng lặp
data_train = pd.read_csv('src/output/data/data_train.csv')

# Tạo bảng thống kê cơ bản
report = data_train.describe().T
report["median"] = data_train.median()
report["dtype"] = data_train.dtypes
report["nan"] = data_train.isnull().sum()

# Làm tròn các giá trị số đến 2 chữ số thập phân
report = report.round(3)

# Reset index để đưa tên đặc trưng thành một cột
report = report.reset_index()
report = report.rename(columns={"index": "feature_name"})

# In báo cáo
print(report)

# Lưu file CSV, có cột 'feature_name'
report.to_csv('src/output/data_analysis/report_data_train.csv', index=False)
