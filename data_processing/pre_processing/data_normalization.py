import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

# Đọc dữ liệu
data = pd.read_csv("data_processing/feature/data_train.csv")
data_test = pd.read_csv("data_processing/feature/data_test.csv")
minMaxScaler = MinMaxScaler()
standardScaler = StandardScaler()
robustScaler = RobustScaler()
# standard scaler train
standard_scaler_train = ['radomain','rapath']
data[standard_scaler_train]=standardScaler.fit_transform(data[standard_scaler_train])
# # Chọn các cột cần log transform
# features_to_transform = ['length', 'domain_len']

# data['length'] = robustScaler.fit_transform(data["length"].values.reshape(-1, 1))
# # Áp dụng log transform (log1p để tránh lỗi log(0))
# for feature in features_to_transform:
#     data[feature] = np.log1p(data[feature])
#     data_test[feature] = np.log1p(data_test[feature])

# data[features_to_transform] = standardScaler.fit_transform(data[features_to_transform])
# data_test[features_to_transform] = standardScaler.transform(data_test[features_to_transform])  # Dùng transform thay vì fit_transform để đảm bảo nhất quán

# Lưu lại dữ liệu đã xử lý
data.to_csv("data_processing/feature/data_train_processed.csv", index=False)
data_test.to_csv("data_processing/feature/data_test_processed.csv", index=False)
