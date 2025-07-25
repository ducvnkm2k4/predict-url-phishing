import pandas as pd
import os
from urllib.parse import urlparse

def valid_url(url):
    try:
        parsed = urlparse(str(url))
        # Có scheme http/https
        if parsed.scheme in ["http", "https"]:
            return True
        # Không có scheme, nhưng netloc hoặc path bắt đầu bằng www.
        if not parsed.scheme:
            if parsed.netloc.startswith("www."):
                return True
            # Trường hợp url kiểu www.example.com (netloc rỗng, path chứa domain)
            if parsed.path.startswith("www."):
                return True
        return False
    except Exception:
        return False

def merge_dataset():
    os.makedirs("src/output/data", exist_ok=True)
    # Đọc dataset 2: 
    # dataset 2 gồm 2 nhãn phishing và legitimate
    data2 = pd.read_csv("src/dataset/data/data2/dataset_phishing.csv")[["url", "status"]]
    data2["label"] = (data2["status"] == "phishing").astype(int)
    data2.drop(columns=["status"], inplace=True)

    # Đọc dataset 3
    # dataset 3 gồm 2 nhãn 1 và 0 (1 cho legitimate và 0 cho phishing)
    data3_feature = pd.read_csv("src/dataset/data/data3/features.csv")
    data3_label = pd.read_csv("src/dataset/data/data3/targets.csv")
    data3 = pd.DataFrame({"url": data3_feature["URL"], "label": data3_label["label"]})
    data3['label'] = (data3["label"]==0).astype(int)

    # Đọc dataset 4
    # dataset 4 gồm 2 nhãn 0 và 1 (0 cho legitimate và 1 cho phishing)
    data4_raw = pd.read_csv('src/dataset/data/data4/urlset.csv', encoding='latin1')
    data4 = pd.DataFrame()
    data4['url'] = data4_raw['domain'].str.strip("'\"")
    data4["label"] = (data4_raw["label"] == 1.0).astype(int)

    # Đọc dataset 6
    # gồm 2 nhãn 0 và 1( 0 cho legitimate 1 cho phishing)
    data6_legitimate = pd.read_csv('src/dataset/data/data6/extracted_legitmate_dataset.csv')
    data6_phishing = pd.read_csv('src/dataset/data/data6/extracted_phishing_dataset.csv')
    data6_legitimate['url'] = data6_legitimate['protocol'] + '://' + data6_legitimate['domain_name'] + '/' + data6_legitimate["address"]
    data6_phishing['url'] = data6_phishing['protocol'] + '://' + data6_phishing['domain_name'] + '/' + data6_phishing["address"]
    data6 = pd.concat([data6_legitimate, data6_phishing], ignore_index=True)
    data6 = data6[['url', 'label']]

    # Đọc dataset 7
    # gồm 2 nhãn 0 và 1 (0 cho legitimate 1 cho phishing)
    data7 = pd.read_csv('src/dataset/data/data7/data_bal - 20000.csv')
    data7 = data7.rename(columns={"Labels": 'label', 'URLs': 'url'})

    # Đọc dataset 8
    # gồm 2 nhãn legitimate và phishing
    data8_test = pd.read_csv('src/dataset/data/data8/test.csv')
    data8_test["label"] = (data8_test["label"] == 'phishing').astype(int)
    data8_train = pd.read_csv('src/dataset/data/data8/train.csv')
    data8_train['label'] = (data8_train['label'] == 'phishing').astype(int)
    data8_val = pd.read_csv('src/dataset/data/data8/val.csv')
    data8_val['label'] = (data8_val['label'] == 'phishing').astype(int)
    # Tạo data_train
    data_train = pd.concat([data2, data3, data4, data6, data8_test], ignore_index=True)
    data_train = data_train.dropna(subset=["url", "label"])
    data_train = data_train[data_train["url"].apply(lambda x: isinstance(x, str) and x.isascii())]
    # Lọc url hợp lệ
    # data_train = data_train[data_train["url"].apply(valid_url)]
    print('-------------------data train--------------------------')
    print(data_train)
    data_train = data_train.drop_duplicates()
    print('-------------------data train delete duplicate--------------------------')
    print(data_train)
    
    # Tạo data_test
    data_test = data7
    data_test = data_test[data_test["url"].apply(lambda x: x.isascii())]
    # data_test = data_test[data_test["url"].apply(valid_url)]
    print('----------------------data test---------------------------')
    print(data_test)
    data_test = data_test.drop_duplicates()
    print('----------------------data test delete duplicate---------------------------')
    print(data_test)
    return [data_train,data_test]

if __name__ == "__main__":
    data_train,data_test= merge_dataset()
    # Lưu dữ liệu đã xử lý
    data_train.to_csv('src/output/data/data_train_raw.csv', index=False)
    data_test.to_csv('src/output/data/data_test_raw.csv', index=False)
