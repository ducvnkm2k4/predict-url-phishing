import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler,QuantileTransformer

def data_normalization(data_train, data_test):

    # --- Hàm kiểm tra nhị phân --- #
    def is_binary(col):
        return set(col.dropna().unique()) <= {0, 1}

    # --- Xác định các cột nhị phân và không nhị phân --- #
    binary_cols = [col for col in data_train.columns if is_binary(data_train[col])]

    log_transform_feature = ['length', 'tachar', 'numdot', 'countUpcase', 'numsdm', 'domain_len', 'ent_char', 'eod']
    standard_feature = ['radomain','rapath']
    # --- Áp dụng log transform cho các feature cần thiết --- #
    for col in log_transform_feature:
        if col in data_train.columns:
            data_train[col] = np.log1p(data_train[col])
            data_test[col] = np.log1p(data_test[col])

    for col in standard_feature:
        if col in data_train.columns:
            scaler = QuantileTransformer()
            data_train[col] = scaler.fit_transform(data_train[[col]])
            data_test[col] = scaler.transform(data_test[[col]])
    # --- Loại bỏ các cột nhị phân khỏi danh sách cần scale --- #

    # --- Áp dụng StandardScaler cho các cột cần scale --- #
    data_train_scaled = data_train.copy()
    data_test_scaled = data_test.copy()
    

    # --- Đảm bảo đúng thứ tự cột --- #
    data_train_scaled = data_train_scaled[data_train.columns]
    data_test_scaled = data_test_scaled[data_test.columns]

    # --- Lưu ra file --- #
    data_train_scaled.to_csv("src/data_processing/feature/data_train_scaled.csv", index=False)
    data_test_scaled.to_csv("src/data_processing/feature/data_test_scaled.csv", index=False)
    return [data_train_scaled, data_test_scaled]


if __name__ == '__main__':
    data_train = pd.read_csv('src/data_processing/feature/data_train.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test.csv')
    data_normalization(data_train, data_test)
