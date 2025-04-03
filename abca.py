import tldextract
from collections import Counter
from urllib.parse import urlparse
import re
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data_train = pd.read_csv('data_processing/raw/data_train_raw.csv')

# Các ký tự đặc biệt % =
#special_chars = "`^&*;@!?#+$|"
special_chars = ""
# Lọc ra các bản ghi có URL chứa ký tự đặc biệt
data_with_special_chars = data_train[data_train['url'].apply(lambda url: any(char in special_chars for char in url))]

# Vẽ histogram theo "label"
plt.figure(figsize=(8, 6))
data_with_special_chars['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])

# Thiết lập các thông số cho biểu đồ
plt.title('Histogram of Labels for URLs Containing Special Characters', fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.show()
