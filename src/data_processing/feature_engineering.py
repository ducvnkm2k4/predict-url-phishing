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
        length,  # 1. length
        sum(1 for c in url if c in special_chars),  # 2. tachar
        int(any(kw in url.lower() for kw in common_keywords)),  # 3. hasKeyWords
        round(sum(len(m) for m in re.findall(hex_pattern, url)) / length, 15),  # 4. tahex
        round(sum(c.isdigit() for c in url) / length, 15),  # 5. tadigit
        url.count('.'),  # 6. numDots
        sum(c.isupper() for c in url),  # 7. countUpcase
        round(sum(c in "aeiou" for c in url.lower()) / length, 15),  # 8. numvo
        round(sum(c.isalpha() and c.lower() not in "aeiou" for c in url) / length, 15),  # 9. numco
        int(any(len(s) > 30 for s in re.findall(r'\S+', url))),  # 10. maxsub30
        round(len(parsed_url.path) / length, 15) if parsed_url.path else 0,  # 11. rapath
        1 if urlparse(url).scheme == "http"  else 0,
        1 if urlparse(url).scheme == "https" else 0,
        1 if parsed_url.netloc.startswith("www.") else 0,
        0 if is_domain_ip else len(extracted.subdomain.split('.')) if extracted.subdomain else 0,  # 15. numsdm
        round(len(domain) / length, 15) if domain else 0,  # 16. radomain
        round(sum(c in "aeiou" for c in domain.lower()) / len(domain), 15) if domain else 0,  # 18. tanv
        round(sum(c.isalpha() and c.lower() not in "aeiou" for c in domain) / len(domain), 15) if domain else 0,  # 19. tanco
        round(sum(c.isdigit() for c in domain) / len(domain), 15) if domain else 0,  # 20. tandi
        round(sum(c in special_chars_domain for c in domain) / len(domain), 15) if domain else 0,  # 21. tansc
        len(domain),  # 23. domain_len
        round(-sum(p * math.log2(p) for p in domain_char_prob.values()), 15) if domain else 0,  # 24. ent_char
        round(sum(domain.count(c) * char_probabilities.get(c, 0) for c in domain) / len(domain), 15) if domain and char_probabilities else 0,  # 25. eod
        0 if is_domain_ip else int(extracted.registered_domain in top_100k_tranco_list),  # 26. rank
        0 if is_domain_ip else int(extracted.suffix in {"com", "net", "org", "edu", "gov"}),  # 27. tld
        0 if is_domain_ip else int(extracted.suffix in {'tk', 'ml', 'cf', 'ga', 'gq', 'xyz', 'top', 'cn', 'ru', 'work', 'club', 'site'}),  # 29. hasSuspiciousTld
        label  # 30. label
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
    "length", "tachar", "hasKeyWords", "tahex", "tadigit", "numDots", "countUpcase",
    "numvo", "numco", "maxsub30", "rapath", 'http','https','www',  "numsdm",
    "radomain", "tanv", "tanco", "tandi", "tansc",  "domain_len",
    "ent_char", "eod", "rank", "tld", "hasSuspiciousTld", "label"
    ]


    data_train_feature = pd.DataFrame(extracted_features_train, columns=feature_names)
    data_test_feature = pd.DataFrame(extracted_features_test, columns=feature_names)

    # ✅ Nếu bạn muốn loại bỏ dòng trùng lặp trong features:
    data_train_feature = data_train_feature.drop_duplicates()
    data_test_feature = data_test_feature.drop_duplicates()

    data_train_feature.to_csv('src/data_processing/feature/data_train.csv',index=None)
    data_test_feature.to_csv('src/data_processing/feature/data_test.csv',index=None)
    print("✅ Trích xuất đặc trưng hoàn thành! 🚀")