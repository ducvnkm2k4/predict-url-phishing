import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler,QuantileTransformer

def data_normalization(data_train, data_test):
    # --- Các nhóm đặc trưng theo loại chuẩn hóa --- #
    log_transform_feature = ['length', 'tachar', 'numDots', 'countUpcase', 'numsdm', 'domain_len', 'ent_char', 'eod']
    quantile_feature = ['radomain', 'rapath']
    min_max_feature = ['tandi', 'tahex', 'tadigit', 'tanco']
    robust_feature = ['numvo', 'numco', 'tanv', 'tansc']  

    # --- Log transform --- #
    for col in log_transform_feature:
        if col in data_train.columns:
            data_train[col] = np.log1p(data_train[col])
            data_test[col] = np.log1p(data_test[col])

    # --- QuantileTransformer --- #
    for col in quantile_feature:
        if col in data_train.columns:
            scaler = QuantileTransformer(output_distribution='normal')
            data_train[col] = scaler.fit_transform(data_train[[col]])
            data_test[col] = scaler.transform(data_test[[col]])

    # --- MinMaxScaler --- #
    for col in min_max_feature:
        if col in data_train.columns:
            scaler = MinMaxScaler()
            data_train[col] = scaler.fit_transform(data_train[[col]])
            data_test[col] = scaler.transform(data_test[[col]])

    # --- RobustScaler --- #
    for col in robust_feature:
        if col in data_train.columns:
            scaler = RobustScaler()
            data_train[col] = scaler.fit_transform(data_train[[col]])
            data_test[col] = scaler.transform(data_test[[col]])

    # --- Ghi lại dữ liệu đã scale --- #
    data_train.to_csv("src/data_processing/feature/data_train_scaled.csv", index=False)
    data_test.to_csv("src/data_processing/feature/data_test_scaled.csv", index=False)

    print("✅ Hoàn tất chuẩn hóa đặc trưng!")
    return data_train, data_test


if __name__ == '__main__':
    data_train = pd.read_csv('src/data_processing/feature/data_train.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test.csv')
    data_normalization(data_train, data_test)
