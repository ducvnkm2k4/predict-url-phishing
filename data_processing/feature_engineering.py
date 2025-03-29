import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Hiển thị tiến trình

# Load dữ liệu
data_train = pd.read_csv('dataset/raw/data_train_raw.csv')
data_test = pd.read_csv('dataset/raw/data_test_raw.csv')

# Đọc danh sách top 100k domain từ Tranco
top_100k_tranco_list = set(pd.read_csv('dataset/tranco_list/tranco_5897N.csv', header=None).iloc[:, 1].tolist())

# Đọc bảng xác suất ký tự (ngram)
char_probabilities = pd.read_csv('dataset/tranco_list/char_probabilities.csv').set_index('Character')['Probability'].to_dict()

# Các danh sách keyword và ký tự đặc biệt
special_chars = "`%^&*;@!?#=+$"
special_chars_domain=".-_"
hex_pattern = re.compile(r'[a-fA-F0-9]{10,}')
common_keywords = {"password", "login", "secure", "account", "index", "token", "signin", "update", "verify", "auth", "security"}
sensitive_keywords = {"confirm", "submit", "payment", "invoice", "billing", "transaction", "transfer", "refund", "wire"}
short_url_services = {"bit.ly", "goo.gl", "tinyurl.com", "is.gd", "t.co", "ow.ly", "cutt.ly", "shrtco.de", "rebrand.ly", "lnkd.in"}
redirect_keywords = {"redirect", "url", "next", "dest", "destination", "forward", "go", "to"}

# **Hàm trích xuất đặc trưng**
def extract_features(params):
    url, label = params  # Nhận tuple (url, label)
    parsed_url = urlparse(url)
    domain = parsed_url.hostname if parsed_url.hostname else ""
    length = len(url)

    # Tính toán các đặc trưng
    f1 = length  
    f2 = sum(1 for char in url if char in special_chars) / length
    f3 = int(any(kw in url.lower() for kw in common_keywords))
    f4 = int(any(char in url for char in special_chars))
    f5 = int(any(kw in url.lower() for kw in sensitive_keywords))
    f6 = int(any(service in url for service in short_url_services))
    f7 = sum(len(match) for match in re.findall(hex_pattern, url)) / length
    f8 = sum(1 for char in url if char.isdigit()) / length
    f9 = url.count('.')
    f10 = url.count('/') / length
    f11 = sum(1 for char in url if char.isupper())
    f12 = sum(1 for char in url if char.lower() in "aeiou") / length
    f13 = sum(1 for char in url if char.isalpha() and char.lower() not in "aeiou") / length
    f14 = domain.count('.') if domain else 0
    f15 = len(domain.split('.')[0]) / length if domain else 0
    f16 = len(parsed_url.path) / length if parsed_url.path else 0
    f17 = int(any(kw in url.lower() for kw in ["http", "https", "www"]))
    f18 = int(re.fullmatch(r'\b\d{1,3}(\.\d{1,3}){3}\b', domain or "") is not None)
    f19 = int(parsed_url.path.lower().endswith(".exe"))
    f20 = int(parsed_url.port is not None)
    f21 = int('\\' in url)
    f22 = int(any(kw in parsed_url.query.lower() for kw in redirect_keywords))
    f23 = int(any(kw in parsed_url.query.lower() for kw in ["ref", "cdm", "track", "utm"]))
    f24 = int(any(len(sub) > 30 for sub in re.findall(r'\S+', url)))

    # Đặc trưng tên miền
    f25 = sum(1 for char in domain if char in "aeiou") / len(domain) if domain else 0
    f26 = sum(1 for char in domain if char.isalpha() and char.lower() not in "aeiou") / len(domain) if domain else 0
    f27 = sum(1 for char in domain if char.isdigit()) / len(domain) if domain else 0
    f28 = sum(1 for char in domain if char in special_chars_domain) / len(domain) if domain else 0
    f29 = sum(len(match) for match in re.findall(hex_pattern, domain)) / len(domain) if domain else 0
    f30 = int(domain[0].isdigit()) if domain else 0
    f31 = len(domain) if domain else 0
    f32 = -sum((domain.count(c) / len(domain)) * np.log2(domain.count(c) / len(domain)) for c in set(domain)) if domain else 0
    f33 = sum(domain.count(c) * char_probabilities.get(c, 0) for c in domain) / len(domain) if domain else 0
    f34 = int(domain in top_100k_tranco_list)
    f35 = int(domain.split('.')[-1] in {"com", "net", "org", "edu", "gov"})

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
            f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, label]  # Gán label cuối cùng

# **Hàm chạy multiprocessing**
def parallel_feature_extraction(url_list, label_list):
    num_cores = cpu_count() - 1  # Dùng (N-1) CPU cores
    print(f"⏳ Đang xử lý với {num_cores} CPU cores...")

    params = list(zip(url_list, label_list))  # Tạo danh sách tuple (url, label)

    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(extract_features, params), total=len(url_list), desc="Extracting Features"))

    return results

if __name__ == "__main__":
    # Chạy feature extraction trên tập train & test
    extracted_features_train = parallel_feature_extraction(data_train['url'], data_train['label'])
    extracted_features_test = parallel_feature_extraction(data_test['url'], data_test['label'])

    # Tạo DataFrame
    feature_names = ["length", "tachar", "hasKeyWords", "hasSpecialChar", "hasspecKW", "tinyUrl", "tahex", "tadigit", "numDots",
                     "taslash", "countUpcase", "numvo", "numco", "numsdm", "radomain", "rapath", "haspro", "hasIP", "hasExe",
                     "hasport", "backslash", "redirect", "hasref", "maxsub30", "tanv", "tanco", "tandi", "tansc", "tanhe",
                     "is_digit", "len", "ent_char", "eod", "rank", "tld", "label"]

    data_train_feature = pd.DataFrame(extracted_features_train, columns=feature_names)
    data_test_feature = pd.DataFrame(extracted_features_test, columns=feature_names)

    # Lưu kết quả
    data_train_feature.to_csv('dataset/feature/data_train.csv', index=False)
    data_test_feature.to_csv('dataset/feature/data_test.csv', index=False)
    print("✅ Trích xuất đặc trưng hoàn thành! 🚀")
