import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas

data = pandas.read_csv("dataset/feature/data_train.csv")
data_test = pandas.read_csv("dataset/feature/data_test.csv")
data['length'] = np.log1p(data['length'])  # log(1+x) để tránh log(0)
data_test['length'] = np.log1p(data_test['length'])
scaler = MinMaxScaler()
data[['length']] = scaler.fit_transform(data[['length']])
data[['domain_len']] = scaler.fit_transform(data[['domain_len']])

data_test[['length']] = scaler.fit_transform(data_test[['length']])
data_test[['domain_len']] = scaler.fit_transform(data_test[['domain_len']])
data.to_csv('dataset/feature/data_train_processed.csv', index=False)
data_test.to_csv('dataset/feature/data_test_processed.csv', index=False)
