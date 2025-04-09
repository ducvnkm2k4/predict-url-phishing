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
import os
from bs4 import BeautifulSoup
import requests
# Load dữ liệu
data_train = pd.read_csv('data_processing/raw/data_train_raw.csv')
data_test = pd.read_csv('data_processing/raw/data_test_raw.csv')

# Đọc danh sách top 100k domain từ Tranco
top_100k_tranco_list = set(pd.read_csv('dataset/tranco_list/tranco_5897N.csv', header=None).iloc[:, 1].tolist())

# Đọc bảng xác suất ký tự (ngram)
char_probabilities = pd.read_csv('dataset/tranco_list/char_probabilities.csv').set_index('Character')['Probability'].to_dict()

# Các danh sách keyword và ký tự đặc biệt
special_chars = "`%^&*;@!?#=+$|"
special_chars_domain=".-_"
hex_pattern = re.compile(r'[a-fA-F0-9]{10,}')
common_keywords = {
    "password", "login", "secure", "account", "index", "token", "signin", 
    "update", "verify", "auth", "security","confirm", "submit", "payment", 
    "invoice", "billing", "transaction", "transfer", "refund", "wire"
    }
# sensitive_keywords = {"confirm", "submit", "payment", "invoice", "billing", "transaction", "transfer", "refund", "wire"}
short_url_services =set( pd.read_csv('dataset/short_url_services.csv').drop_duplicates().iloc[:,0])
redirect_keywords = {"redirect=", "url=", "next=", "dest=", "destination=", "forward=", "go=", "to="}

ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')

# **Hàm trích xuất đặc trưng**
def extract_features(params):
    url, label = params  # Nhận tuple (url, label)
    url=url.strip("'\"")
    parsed_url = urlparse(url)

    url_for_parsing = url if parsed_url.scheme in {"http","https"} else "http://"+url
    parsed_url=urlparse(url_for_parsing)
    extracted = tldextract.extract(url)
    domain = parsed_url.hostname
    length = len(url)
    is_domain_ip = re.fullmatch(ip_pattern, domain)
    char_counts = Counter(domain)
    total_chars = sum(char_counts.values())
    domain_char_probabilities = {char: count / total_chars for char, count in char_counts.items()}
    
    # Tính toán các đặc trưng trên url
    # 1.length: độ dài URL
    length = length  
    # 2.tachar: tỷ lệ ký tự đặc biệt trong URL
    tachar = sum(1 for char in url if char in special_chars)
    # hasKeyWords: URL chứa từ khóa phổ biến
    hasKeyWords = int(any(kw in url.lower() for kw in common_keywords)) 
    # hasspecKW: URL chứa từ khóa nhạy cảm
    # hasspecKW = int(any(kw in url.lower() for kw in sensitive_keywords)) 
    # tahex: tỷ lệ chuỗi hex trong URL
    tahex = round(sum(len(match) for match in re.findall(hex_pattern, url)) / length,15) 
    # tadigit: tỷ lệ chữ số trong URL 
    tadigit = round(sum(1 for char in url if char.isdigit()) / length ,15) 
    # numDots: số dấu chấm trong URL
    numDots = url.count('.')
    # countUpcase: số ký tự in hoa trong URL
    countUpcase = sum(1 for char in url if char.isupper())
    # numvo: tỷ lệ nguyên âm trong URL
    numvo = round(sum(1 for char in url if char.lower() in "aeiou") / length,15)
    # numco: tỷ lệ phụ âm trong URL
    numco = round(sum(1 for char in url if char.isalpha() and char.lower() not in "aeiou") / length,15)  
    # maxsub30: URL chứa chuỗi con dài >30 ký tự
    maxsub30 = int(any(len(sub) > 30 for sub in re.findall(r'\S+', url))) 
    # rapath: độ dài đường dẫn so với toàn bộ URL 
    rapath = round(len(parsed_url.path) / length,15) if parsed_url.path else 0 
    # haspro: có chứa http, https, www hay không
    haspro = 1 if urlparse(url).scheme in {"http", "https"} or parsed_url.netloc.startswith("www.") else 0
    # hasref: URL chứa tham số theo dõi
    # hasref = int(any(kw in parsed_url.query.lower() for kw in ["ref=", "cdm=", "track=", "utm="]) 
    #             and "href=" not in parsed_url.query.lower() 
    #             and "notrack=1" not in parsed_url.query.lower())

    # Đặc trưng tên miền
    # hasIP: URL chứa địa chỉ IP
    # hasIP = int(is_domain_ip is not None)  
    # # hasport: URL có chứa số cổng
    # hasport = int(parsed_url.port is not None)  
    # numsdm: số lượng subdomain trong tên miền
    numsdm = 0 if is_domain_ip else domain.count('.') - 1 
    # radomain: tỷ lệ độ dài của domain so với tên miền
    radomain = round(len(domain) / length if domain else 0,15)  
    # tinyUrl: URL là dịch vụ rút gọn
    tinyUrl= int(domain in short_url_services)  
    # tanv: tỷ lệ nguyên âm trong tên miền
    tanv = round(sum(1 for char in domain if char in "aeiou") / len(domain),15) if domain else 0  
    # tanco: tỷ lệ phụ âm trong tên miền
    tanco = round(sum(1 for char in domain if char.isalpha() and char.lower() not in "aeiou") / len(domain),15) if domain else 0  
    # tandi: tỷ lệ chữ số trong tên miền
    tandi = round(sum(1 for char in domain if char.isdigit()) / len(domain),15) if domain else 0 
    # tansc: tỷ lệ ký tự đặc biệt trong tên miền
    tansc = round(sum(1 for char in domain if char in special_chars_domain) / len(domain),15) if domain else 0  
    # is_digit: tên miền bắt đầu bằng số
    # is_digit = int(domain[0].isdigit()) if domain else 0  
    # len: độ dài tên miền
    domain_length = len(domain) if domain else 0  
    # ent_char: entropy của ký tự trong tên miền
    ent_char = round(-sum(p * math.log2(p) for p in domain_char_probabilities.values()),15) if domain else 0 
    # eod của tên miền
    eod = round(sum(domain.count(c) * char_probabilities.get(c, 0) for c in domain) / len(domain),15) if domain and char_probabilities else 0
    # rank: tên miền thuộc top 100k của Tranco
    rank = 0 if is_domain_ip else int( extracted.registered_domain in top_100k_tranco_list) 
    # tld: tên miền thuộc TLD phổ biến
    tld = 0 if is_domain_ip else int(extracted.suffix in {"com", "net", "org", "edu", "gov"})  
    # hasdoubleslash=1 if url.count('//') - 1 > 1 else 0
    # hasSuspiciousTld: một số tld phổ biến của url phishing
    hasSuspiciousTld =0 if is_domain_ip else int( extracted.suffix in {'tk', 'ml', 'cf', 'ga', 'gq'})

    # return [
    #     length, tachar, hasKeyWords, hasspecKW, tahex, 
    #     tadigit, numDots, taslash, countUpcase, numvo, numco, 
    #     maxsub30, rapath, haspro, hasref, 
    #     hasIP, hasport, numsdm, radomain, tinyUrl, tanv, 
    #     tanco, tandi, tansc, is_digit, 
    #     domain_length, ent_char, eod, rank, tld,
    #     tld_phishing, label]

    return [
        length, tachar, hasKeyWords, tahex, 
        tadigit, numDots, countUpcase, numvo, numco, 
        maxsub30, rapath, haspro, 
         numsdm, radomain, tinyUrl, tanv, 
        tanco, tandi, tansc,
        domain_length, ent_char, eod, rank, tld,
         hasSuspiciousTld, label]

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

    # feature_names = [
    #      
    #        
    #    
    #      
    #     
    #     "eod",
    #     
    #     ]

    feature_names = [
        "length", "tachar", "hasKeyWords","tahex", 
        "tadigit", "numDots", "countUpcase", "numvo", "numco",
        "maxsub30", "rapath","haspro",
        "numsdm", "radomain","tinyUrl", "tanv", 
        "tanco", "tandi", "tansc",
        "domain_len", "ent_char", "eod", "rank", "tld",
        "hasSuspiciousTld", "label"
        ]


    data_train_feature = pd.DataFrame(extracted_features_train, columns=feature_names)
    data_test_feature = pd.DataFrame(extracted_features_test, columns=feature_names)

    delete_duplicate(data_train=data_train_feature,data_test=data_test_feature)

    print("✅ Trích xuất đặc trưng hoàn thành! 🚀")
