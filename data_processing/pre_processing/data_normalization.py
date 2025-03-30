import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas

data = pandas.read_csv("dataset/feature/data_train.csv")
data_test = pandas.read_csv("dataset/feature/data_test.csv")
# đặc trưng có chứa out
feature_outliers=[]
# phân phối lệch
feature_skewness=['length']

data.to_csv('dataset/feature/data_train_processed.csv', index=False)
data_test.to_csv('dataset/feature/data_test_processed.csv', index=False)
