import pandas as pd

# Đọc dữ liệu sau khi loại bỏ trùng lặp
data_train = pd.read_csv('data_processing/feature/data_train_processed.csv')

# Tạo bảng thống kê cơ bản
report = data_train.describe().T  # Xoay bảng để hiển thị theo cột
report["median"] = data_train.median()  # Tính trung vị (median)
report["dtype"] = data_train.dtypes  # Thêm cột kiểu dữ liệu
report['nan']=data_train.isnull().sum()
# In báo cáo
print(report)
#report.to_csv('report_dataset/report_data.csv',index=None)

