import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tldextract

# Đọc dữ liệu
df = pd.read_csv('src/data_processing/raw/data_train_raw.csv')

# Tính đặc trưng có hậu tố nghi ngờ (TLD)
suspicious_tlds = {'tk', 'ml', 'cf', 'ga', 'gq', 'xyz', 'top', 'cn', 'ru', 'work', 'club', 'site'}
df['has_suspicious_suffix'] = df['url'].apply(
    lambda url: int(tldextract.extract(url).suffix in suspicious_tlds)
)

# Vẽ biểu đồ
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='has_suspicious_suffix', hue='label')
plt.title("Phân bố đặc trưng 'has_suspicious_suffix' theo nhãn")
plt.xlabel("has_suspicious_suffix (1 = Có TLD nghi ngờ, 0 = Không)")
plt.ylabel("Số lượng URL")

# Ghi số lượng lên cột
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height() + 3,
             f'{int(p.get_height())}', ha='center')

plt.tight_layout()
plt.show()
