import pandas as pd
from collections import Counter

# Đọc dữ liệu từ file CSV
top_100k_tranco = pd.read_csv('dataset/tranco_list/tranco_5897N.csv')

# Giả sử danh sách domain nằm ở cột đầu tiên
domains = top_100k_tranco.iloc[:, 1].astype(str)  # Chuyển tất cả về string

# Hàm tạo bi-gram từ chuỗi
def generate_bigrams(domain):
    return [domain[i:i+3] for i in range(len(domain)-1)]

# Tạo danh sách tất cả các bi-gram
all_bigrams = []
for domain in domains:
    all_bigrams.extend(generate_bigrams(domain))

# Đếm tần suất xuất hiện của các bi-gram
bigram_counts = Counter(all_bigrams)
print(bigram_counts)
# Chuyển sang DataFrame và sắp xếp theo tần suất
bigram_df = pd.DataFrame(bigram_counts.items(), columns=['Bi-gram', 'Frequency'])
bigram_df = bigram_df.sort_values(by='Frequency', ascending=False)

# Lấy top 1000 bi-gram phổ biến nhất
top_1000_bigrams = bigram_df

# Hiển thị top 10 bi-gram phổ biến nhất
print(top_1000_bigrams)

# Lưu kết quả vào file CSV nếu cần
top_1000_bigrams.to_csv('data_processing/n_gram_pro/trigrams.csv', index=False)
