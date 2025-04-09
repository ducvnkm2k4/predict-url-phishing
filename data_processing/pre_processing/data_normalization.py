from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

data_train = pd.read_csv('data_processing/feature/data_train.csv')
data_test = pd.read_csv('data_processing/feature/data_test.csv')

# Hàm kiểm tra một cột có phải nhị phân không
def is_binary(col):
    return set(col.unique()) <= {0, 1}

# Lọc các cột nhị phân từ tập train
binary_cols = [col for col in data_train.columns if is_binary(data_train[col])]

# Tạo DataFrame chỉ chứa các cột nhị phân
data_train_bit = data_train[binary_cols]
data_test_bit = data_test[binary_cols]  # Dùng cùng tên cột

log_and_trim_cols = ['length', 'tachar', 'tahex', 'tadigit', 'numDots', 'countUpcase', 'domain_len', 'numsdm','eod']
standard_only_cols = ['numvo', 'numco', 'tanv', 'tanco', 'rapath', 'tandi', 'tansc', 'radomain', 'ent_char']

# --- Log-transform custom transformer --- #
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

# --- IQR Trimmer --- #
class IQRTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        self.bounds = {}
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = (Q1 - self.factor * IQR, Q3 + self.factor * IQR)
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in X.columns:
            lower, upper = self.bounds[col]
            X_[col] = X_[col].clip(lower, upper)
        return X_

# --- Pipeline cho các cột lệch & nhiều outliers --- #
log_trim_pipeline = Pipeline([
    ('trim', IQRTrimmer()),
    ('log', LogTransformer()),
    ('scale', StandardScaler())
])

# --- Pipeline cho các cột gần chuẩn --- #
standard_pipeline = Pipeline([
    ('scale', StandardScaler())
])

# --- Gộp lại thành ColumnTransformer --- #
full_pipeline = ColumnTransformer([
    ('log_trim', log_trim_pipeline, log_and_trim_cols),
    ('standard', standard_pipeline, standard_only_cols)
])

# Fit trên train
X_train_processed = full_pipeline.fit_transform(data_train)

# Transform trên test
X_test_processed = full_pipeline.transform(data_test)

# Kết hợp danh sách tên cột đã xử lý
all_columns = log_and_trim_cols + standard_only_cols

# Trả về DataFrame để dễ dùng tiếp
X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_columns)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_columns)



data_train_scaled = pd.concat([X_train_processed_df, data_train_bit], axis=1)
data_test_scaled = pd.concat([X_test_processed_df, data_test_bit],axis=1)

print(data_test_scaled.head())
data_train_scaled.to_csv("data_processing/feature/data_train_scaled.csv", index=False)
data_test_scaled.to_csv("data_processing/feature/data_test_scaled.csv", index=False)
