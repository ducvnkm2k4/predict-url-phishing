from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd

# Chỉ chọn các cột số
data= pd.read_csv('dataset/feature/data_train.csv')
X = data.select_dtypes(include=[np.number])

# Tính VIF cho từng biến
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# In kết quả
print(vif_data.sort_values(by="VIF", ascending=False))
