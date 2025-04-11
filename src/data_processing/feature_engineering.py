import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
import tldextract
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter
import math

# Biến toàn cục cho multiprocessing
char_probabilities = {}
top_100k_tranco_list = set()

# Các pattern và cấu hình
special_chars = "`%^&*;@!?#=+$|"
special_chars_domain = ".-_"
hex_pattern = re.compile(r'[a-fA-F0-9]{10,}')
common_keywords = {
    "password", "login", "secure", "account", "index", "token", "signin", 
    "update", "verify", "auth", "security", "confirm", "submit", "payment", 
    "invoice", "billing", "transaction", "transfer", "refund", "wire"
}
short_url_services = set(pd.read_csv('src/dataset/short_url_services.csv').drop_duplicates().iloc[:, 0])
redirect_keywords = {"redirect=", "url=", "next=", "dest=", "destination=", "forward=", "go=", "to="}
ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')


# ⭐ Init worker cho multiprocessing
def init_worker(char_probs, top_domains):
    global char_probabilities
    global top_100k_tranco_list
    char_probabilities = char_probs
    top_100k_tranco_list = top_domains


# ⭐ Hàm trích xuất đặc trưng
def extract_features(params):
    url, label = params
    url = url.strip("'\"")
    parsed_url = urlparse(url if urlparse(url).scheme else "http://" + url)
    extracted = tldextract.extract(url)
    domain = parsed_url.hostname or ""
    length = len(url)
    is_domain_ip = re.fullmatch(ip_pattern, domain)
    char_counts = Counter(domain)
    total_chars = sum(char_counts.values())
    domain_char_prob = {char: count / total_chars for char, count in char_counts.items()}

    features = [
        length,
        sum(1 for c in url if c in special_chars),  # tachar
        int(any(kw in url.lower() for kw in common_keywords)),  # hasKeyWords
        round(sum(len(m) for m in re.findall(hex_pattern, url)) / length, 15),  # tahex
        round(sum(c.isdigit() for c in url) / length, 15),  # tadigit
        url.count('.'),  # numDots
        sum(c.isupper() for c in url),  # countUpcase
        round(sum(c in "aeiou" for c in url.lower()) / length, 15),  # numvo
        round(sum(c.isalpha() and c.lower() not in "aeiou" for c in url) / length, 15),  # numco
        int(any(len(s) > 30 for s in re.findall(r'\S+', url))),  # maxsub30
        round(len(parsed_url.path) / length, 15) if parsed_url.path else 0,  # rapath
        int(parsed_url.scheme in {"http", "https"} or parsed_url.netloc.startswith("www.")),  # haspro
        0 if is_domain_ip else domain.count('.') - 1,  # numsdm
        round(len(domain) / length, 15) if domain else 0,  # radomain
        int(domain in short_url_services),  # tinyUrl
        round(sum(c in "aeiou" for c in domain.lower()) / len(domain), 15) if domain else 0,  # tanv
        round(sum(c.isalpha() and c.lower() not in "aeiou" for c in domain) / len(domain), 15) if domain else 0,  # tanco
        round(sum(c.isdigit() for c in domain) / len(domain), 15) if domain else 0,  # tandi
        round(sum(c in special_chars_domain for c in domain) / len(domain), 15) if domain else 0,  # tansc
        len(domain),  # domain_len
        round(-sum(p * math.log2(p) for p in domain_char_prob.values()), 15) if domain else 0,  # ent_char
        round(sum(domain.count(c) * char_probabilities.get(c, 0) for c in domain) / len(domain), 15) if domain and char_probabilities else 0,  # eod
        0 if is_domain_ip else int(extracted.registered_domain in top_100k_tranco_list),  # rank
        0 if is_domain_ip else int(extracted.suffix in {"com", "net", "org", "edu", "gov"}),  # tld
        0 if is_domain_ip else int(extracted.suffix in {'tk', 'ml', 'cf', 'ga', 'gq'}),  # hasSuspiciousTld
        label
    ]

    return features


# ⭐ Hàm chạy multiprocessing
def parallel_feature_extraction(url_list, label_list, char_probs, top_domains):
    num_cores = cpu_count()
    print(f"⏳ Đang xử lý với {num_cores} CPU cores...")

    params = list(zip(url_list, label_list))
    with Pool(num_cores, initializer=init_worker, initargs=(char_probs, top_domains)) as pool:
        results = list(tqdm(pool.imap(extract_features, params), total=len(url_list), desc="Extracting Features"))

    return results


if __name__ == "__main__":
    # Load dữ liệu
    data_train = pd.read_csv('src/data_processing/raw/data_train_raw.csv')
    data_test = pd.read_csv('src/data_processing/raw/data_test_raw.csv')

    # Đọc danh sách top 100k domain từ Tranco
    top_100k_tranco_list = set(pd.read_csv('src/dataset/tranco_list/tranco_5897N.csv', header=None).iloc[:, 1].tolist())

    # Đọc bảng xác suất ký tự
    char_probabilities = pd.read_csv('src/dataset/tranco_list/char_probabilities.csv').set_index('Character')['Probability'].to_dict()

    # Trích xuất đặc trưng
    extracted_features_train = parallel_feature_extraction(data_train['url'], data_train['label'], char_probabilities, top_100k_tranco_list)
    extracted_features_test = parallel_feature_extraction(data_test['url'], data_test['label'], char_probabilities, top_100k_tranco_list)

    feature_names = [
        "length", "tachar", "hasKeyWords", "tahex", 
        "tadigit", "numDots", "countUpcase", "numvo", "numco",
        "maxsub30", "rapath", "haspro",
        "numsdm", "radomain", "tinyUrl", "tanv", 
        "tanco", "tandi", "tansc",
        "domain_len", "ent_char", "eod", "rank", "tld",
        "hasSuspiciousTld", "label"
    ]

    data_train_feature = pd.DataFrame(extracted_features_train, columns=feature_names)
    data_test_feature = pd.DataFrame(extracted_features_test, columns=feature_names)

    data_train_feature=data_train.drop_duplicates()
    data_test_feature=data_test.drop_duplicates()

    print("✅ Trích xuất đặc trưng hoàn thành! 🚀")



# # Tính toán các đặc trưng trên url
# # 1.length: độ dài URL
# length = length  
# # 2.tachar: tỷ lệ ký tự đặc biệt trong URL
# tachar = sum(1 for char in url if char in special_chars)
# # 3.hasKeyWords: URL chứa từ khóa phổ biến
# hasKeyWords = int(any(kw in url.lower() for kw in common_keywords)) 
# # 4.hasspecKW: URL chứa từ khóa nhạy cảm
# # hasspecKW = int(any(kw in url.lower() for kw in sensitive_keywords)) 
# # 5.tahex: tỷ lệ chuỗi hex trong URL
# tahex = round(sum(len(match) for match in re.findall(hex_pattern, url)) / length,15) 
# # 6.tadigit: tỷ lệ chữ số trong URL 
# tadigit = round(sum(1 for char in url if char.isdigit()) / length ,15) 
# # 7.numDots: số dấu chấm trong URL
# numDots = url.count('.')
# # 8.countUpcase: số ký tự in hoa trong URL
# countUpcase = sum(1 for char in url if char.isupper())
# # 9.numvo: tỷ lệ nguyên âm trong URL
# numvo = round(sum(1 for char in url if char.lower() in "aeiou") / length,15)
# # 10.numco: tỷ lệ phụ âm trong URL
# numco = round(sum(1 for char in url if char.isalpha() and char.lower() not in "aeiou") / length,15)  
# # 11.maxsub30: URL chứa chuỗi con dài >30 ký tự
# maxsub30 = int(any(len(sub) > 30 for sub in re.findall(r'\S+', url))) 
# # 12.rapath: độ dài đường dẫn so với toàn bộ URL 
# rapath = round(len(parsed_url.path) / length,15) if parsed_url.path else 0 
# # 13.haspro: có chứa http, https, www hay không
# haspro = 1 if urlparse(url).scheme in {"http", "https"} or parsed_url.netloc.startswith("www.") else 0
# # 14.hasref: URL chứa tham số theo dõi
# # hasref = int(any(kw in parsed_url.query.lower() for kw in ["ref=", "cdm=", "track=", "utm="]) 
# #             and "href=" not in parsed_url.query.lower() 
# #             and "notrack=1" not in parsed_url.query.lower())

# # Đặc trưng tên miền
# # 15.hasIP: URL chứa địa chỉ IP
# # hasIP = int(is_domain_ip is not None)  
# # # 16.hasport: URL có chứa số cổng
# # hasport = int(parsed_url.port is not None)  
# # 17.numsdm: số lượng subdomain trong tên miền
# numsdm = 0 if is_domain_ip else domain.count('.') - 1 
# # 18.radomain: tỷ lệ độ dài của domain so với tên miền
# radomain = round(len(domain) / length if domain else 0,15)  
# # 19.tinyUrl: URL là dịch vụ rút gọn
# tinyUrl= int(domain in short_url_services)  
# # 20.tanv: tỷ lệ nguyên âm trong tên miền
# tanv = round(sum(1 for char in domain if char in "aeiou") / len(domain),15) if domain else 0  
# # 21.tanco: tỷ lệ phụ âm trong tên miền
# tanco = round(sum(1 for char in domain if char.isalpha() and char.lower() not in "aeiou") / len(domain),15) if domain else 0  
# # 22.tandi: tỷ lệ chữ số trong tên miền
# tandi = round(sum(1 for char in domain if char.isdigit()) / len(domain),15) if domain else 0 
# # 23.tansc: tỷ lệ ký tự đặc biệt trong tên miền
# tansc = round(sum(1 for char in domain if char in special_chars_domain) / len(domain),15) if domain else 0  
# # 24.is_digit: tên miền bắt đầu bằng số
# # is_digit = int(domain[0].isdigit()) if domain else 0  
# # 25.len: độ dài tên miền
# domain_length = len(domain) if domain else 0  
# # 26.ent_char: entropy của ký tự trong tên miền
# ent_char = round(-sum(p * math.log2(p) for p in domain_char_probabilities.values()),15) if domain else 0 
# # 27.eod của tên miền
# eod = round(sum(domain.count(c) * char_probabilities.get(c, 0) for c in domain) / len(domain),15) if domain and char_probabilities else 0
# # 28.rank: tên miền thuộc top 100k của Tranco
# rank = 0 if is_domain_ip else int( extracted.registered_domain in top_100k_tranco_list) 
# # 29.tld: tên miền thuộc TLD phổ biến
# tld = 0 if is_domain_ip else int(extracted.suffix in {"com", "net", "org", "edu", "gov"})  
# # 30.hasdoubleslash=1 if url.count('//') - 1 > 1 else 0
# # 31:hasSuspiciousTld: một số tld phổ biến của url phishing
# hasSuspiciousTld =0 if is_domain_ip else int( extracted.suffix in {'tk', 'ml', 'cf', 'ga', 'gq'})