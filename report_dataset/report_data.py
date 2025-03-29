import pandas as pd

# Đọc dữ liệu sau khi loại bỏ trùng lặp
df = pd.read_csv('dataset/feature/data_train.csv')

# Tạo bảng thống kê cơ bản
report = df.describe().T  # Xoay bảng để hiển thị theo cột
report["median"] = df.median()  # Tính trung vị (median)
report["dtype"] = df.dtypes  # Thêm cột kiểu dữ liệu

# In báo cáo
print(report)
report.to_csv('report_dataset/report_data.csv',index=None)

