import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler,StandardScaler

def data_normalization(data_train,data_test):
    roubust_scaler= RobustScaler()
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # --- Hàm kiểm tra nhị phân --- #
    def is_binary(col):
        return set(col.dropna().unique()) <= {0, 1}

    # --- Xác định các cột nhị phân và không nhị phân --- #
    binary_cols = [col for col in data_train.columns if is_binary(data_train[col])]
    non_binary_cols = [col for col in data_train.columns if col not in binary_cols]

    log_transform_feature = ['length', 'tachar', 'numdot','countUpcase','numsdm','domain_len','ent_char','eod']
    # min_max_feature = ['tahex','tadigit','rapath','radomain','tandi','tansc']

    # --- Đảm bảo đúng thứ tự cột --- #
    data_train_scaled = data_train_scaled[data_train.columns]
    data_test_scaled = data_test_scaled[data_test.columns]

    # --- Lưu ra file --- #
    data_train_scaled.to_csv("src/data_processing/feature/data_train_scaled.csv", index=False)
    data_test_scaled.to_csv("src/data_processing/feature/data_test_scaled.csv", index=False)
    return [data_train_scaled,data_test_scaled]