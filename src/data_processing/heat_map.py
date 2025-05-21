import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv('src/output/data/data_train.csv')

# Tính ma trận tương quan
correlation_matrix = data.corr()

# Vẽ heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap của ma trận tương quan")
plt.show()
