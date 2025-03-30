import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Hiển thị tiến trình
from collections import Counter
import math
from pre_processing.delete_duplicate import delete_duplicate
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
redirect_keywords = {"redirect=", "url=", "next=", "dest=", "destination=", "forward=", "go=", "to="}
ipv6_pattern = re.compile(r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$')
ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')

# **Hàm trích xuất đặc trưng**
def extract_features(params):
    url, label = params  # Nhận tuple (url, label)
    url=url.strip("'\"")
    parsed_url = urlparse(url)

    url_for_parsing = url if parsed_url.scheme in {"http","https"} else "http://"+url
    parsed_url=urlparse(url_for_parsing)

    domain = parsed_url.hostname
    extracted = tldextract.extract(url)
    domain_extracted = extracted.registered_domain
    length = len(url)

    char_counts = Counter(domain)

    total_chars = sum(char_counts.values())
    domain_char_probabilities = {char: count / total_chars for char, count in char_counts.items()}

    # Tính toán các đặc trưng trên url
    f1 = length  # length: độ dài URL
    f2 = sum(1 for char in url if char in special_chars) / length  # tachar: tỷ lệ ký tự đặc biệt trong URL
    f3 = int(any(kw in url.lower() for kw in common_keywords))  # hasKeyWords: URL chứa từ khóa phổ biến
    f4 = int(any(kw in url.lower() for kw in sensitive_keywords))  # hasspecKW: URL chứa từ khóa nhạy cảm
    f5 = sum(len(match) for match in re.findall(hex_pattern, url)) / length  # tahex: tỷ lệ chuỗi hex trong URL
    f6 = sum(1 for char in url if char.isdigit()) / length  # tadigit: tỷ lệ chữ số trong URL
    f7 = url.count('.')  # numDots: số dấu chấm trong URL
    f8 = url.count('/') / length  # taslash: tỷ lệ dấu '/' trong URL
    f9 = sum(1 for char in url if char.isupper())  # countUpcase: số ký tự in hoa trong URL
    f10 = sum(1 for char in url if char.lower() in "aeiou") / length  # numvo: tỷ lệ nguyên âm trong URL
    f11 = sum(1 for char in url if char.isalpha() and char.lower() not in "aeiou") / length  # numco: tỷ lệ phụ âm trong URL
    f12 = int('\\' in url)  # backslash: URL chứa ký tự '\'
    f13 = int(any(len(sub) > 30 for sub in re.findall(r'\S+', url)))  # maxsub30: URL chứa chuỗi con dài >30 ký tự

    f14 = len(parsed_url.path) / length if parsed_url.path else 0  # rapath: độ dài đường dẫn so với toàn bộ URL
    f15 = 1 if urlparse(url).scheme in {"http", "https"} or parsed_url.netloc.startswith("www.") else 0 # haspro 
    f16 = int(parsed_url.path.lower().endswith(".exe"))  # hasExe: URL kết thúc bằng .exe
    f17 = int(any(kw in parsed_url.query.lower() for kw in redirect_keywords))  # redirect: URL chứa từ khóa điều hướng
    f18 = int(any(kw in parsed_url.query.lower() for kw in ["ref=", "cdm=", "track=", "utm="]) and "href=" not in parsed_url.query.lower() and "notrack=1" not in parsed_url.query.lower())# hasref: URL chứa tham số theo dõi

    is_domain_ip = re.fullmatch(ip_pattern, domain)
    # Đặc trưng tên miền
    f19 = int(is_domain_ip is not None)  # hasIP: URL chứa địa chỉ IP
    f20 = int(parsed_url.port is not None)  # hasport: URL có chứa số cổng
    f21 = 0 if is_domain_ip else domain.count('.') - 1 # numsdm: số lượng subdomain trong tên miền
    f22 = len(domain) / length if domain else 0  # radomain: tỷ lệ độ dài của domain so với tên miền
    f23= int(domain in short_url_services)  # tinyUrl: URL là dịch vụ rút gọn
    f24 = sum(1 for char in domain if char in "aeiou") / len(domain) if domain else 0  # tanv: tỷ lệ nguyên âm trong tên miền
    f25 = sum(1 for char in domain if char.isalpha() and char.lower() not in "aeiou") / len(domain) if domain else 0  # tanco: tỷ lệ phụ âm trong tên miền
    f26 = sum(1 for char in domain if char.isdigit()) / len(domain) if domain else 0  # tandi: tỷ lệ chữ số trong tên miền
    f27 = sum(1 for char in domain if char in special_chars_domain) / len(domain) if domain else 0  # tansc: tỷ lệ ký tự đặc biệt trong tên miền
    f28 = sum(len(match) for match in re.findall(hex_pattern, domain)) / len(domain) if domain else 0  # tanhe: tỷ lệ chuỗi hex trong tên miền
    f29 = int(domain[0].isdigit()) if domain else 0  # is_digit: tên miền bắt đầu bằng số
    f30 = len(domain) if domain else 0  # len: độ dài tên miền

    f31 = -sum((p * math.log(p))/math.log(len(domain)) for p in domain_char_probabilities.values()) if domain else 0  # ent_char: entropy của ký tự trong tên miền
    f32 = sum(domain.count(c) * char_probabilities.get(c, 0) for c in domain) / len(domain) if domain and char_probabilities else 0
    
    f33 = 0 if is_domain_ip else int( domain_extracted in top_100k_tranco_list)  # rank: tên miền thuộc top 100k của Tranco
    f34 = 0 if is_domain_ip else int(extracted.suffix in {"com", "net", "org", "edu", "gov"})  # tld: tên miền thuộc TLD phổ biến

    return [f1, f2, f3,f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
            f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, label]  # Gán label cuối cùng

# **Hàm chạy multiprocessing**
def parallel_feature_extraction(url_list, label_list):
    num_cores = cpu_count()  # Dùng (N-1) CPU cores
    print(f"⏳ Đang xử lý với {num_cores} CPU cores...")

    params = list(zip(url_list, label_list))  # Tạo danh sách tuple (url, label)

    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(extract_features, params), total=len(url_list), desc="Extracting Features"))

    return results

if __name__ == "__main__":
    # Chạy feature extraction trên tập train & test
    extracted_features_train = parallel_feature_extraction(data_train['url'], data_train['label'])
    extracted_features_test = parallel_feature_extraction(data_test['url'], data_test['label'])

    feature_names = ["length", "tachar", "hasKeyWords", "hasspecKW", "tahex", "tadigit", "numDots","taslash", "countUpcase", "numvo", "numco", "backslash", "maxsub30", "rapath","haspro", "hasExe", "redirect", "hasref", "hasIP", "hasport", "numsdm", "radomain","tinyUrl", "tanv", "tanco", "tandi", "tansc", "tanhe", "is_digit", "domain_len", "ent_char", "eod", "rank", "tld", "label"]


    data_train_feature = pd.DataFrame(extracted_features_train, columns=feature_names)
    data_test_feature = pd.DataFrame(extracted_features_test, columns=feature_names)

    delete_duplicate(data_train=data_train_feature,data_test=data_test_feature)
    # # Lưu kết quả
    # data_train_feature.to_csv('dataset/feature/data_train.csv', index=False)
    # data_test_feature.to_csv('dataset/feature/data_test.csv', index=False)
    print("✅ Trích xuất đặc trưng hoàn thành! 🚀")
