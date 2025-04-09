import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse

# Đọc dữ liệu từ file CSV
data_train = pd.read_csv('data_processing/raw/data_train_raw.csv')
common_keywords = {"password", "login", "secure", "account", "index", "token", "signin", "update", "verify", "auth", "security"}
# Hàm kiểm tra xem URL có chứa port hay không
def has_port(url):

    try:
        return any(kw in url.lower() for kw in common_keywords)
    except:
        return False  # Trường hợp URL không hợp lệ

# Lọc ra các bản ghi có URL chứa port
data_with_port = data_train[data_train['url'].apply(has_port)]

# Hiển thị một vài dòng đầu để kiểm tra
print("Số lượng URL chứa port:", len(data_with_port))
print(data_with_port[['url', 'label']].head())

# Vẽ biểu đồ histogram theo 'label'
plt.figure(figsize=(8, 6))
data_with_port['label'].value_counts().plot(kind='bar', color=['lightgreen', 'tomato'])

# Thiết lập các thông số cho biểu đồ
plt.title('Histogram of Labels for URLs Containing Port', fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
 