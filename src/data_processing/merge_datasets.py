import pandas as pd
import os

def merge_dataset():
    os.makedirs("src/data_processing/raw", exist_ok=True)
    # data1=pd.read_csv("dataset/data/data1/malicious_phish.csv")
    # data1["label"]=1
    # data1.loc[data1['type']!="benign","label"] = 0
    # del data1["type"]

    # Đọc dataset 2
    data2 = pd.read_csv("src/dataset/data/data2/dataset_phishing.csv")[["url", "status"]]
    data2["label"] = (data2["status"] == "legitimate").astype(int)
    data2.drop(columns=["status"], inplace=True)

    # Đọc dataset 3
    data3_feature = pd.read_csv("src/dataset/data/data3/features.csv")
    data3_label = pd.read_csv("src/dataset/data/data3/targets.csv")
    data3 = pd.DataFrame({"url": data3_feature["URL"], "label": data3_label["label"]})

    # Đọc dataset 4
    data4_raw = pd.read_csv('src/dataset/data/data4/urlset.csv', encoding='latin1')
    data4 = pd.DataFrame()
    data4['url'] = data4_raw['domain'].str.strip("'\"")
    data4["label"] = (data4_raw["label"] == 0.0).astype(int)

    # Đọc dataset 6
    data6_legitimate = pd.read_csv('src/dataset/data/data6/extracted_legitmate_dataset.csv')
    data6_phishing = pd.read_csv('src/dataset/data/data6/extracted_phishing_dataset.csv')
    data6_legitimate['url'] = data6_legitimate['protocol'] + '://' + data6_legitimate['domain_name'] + '/' + data6_legitimate["address"]
    data6_phishing['url'] = data6_phishing['protocol'] + '://' + data6_phishing['domain_name'] + '/' + data6_phishing["address"]
    data6 = pd.concat([data6_legitimate, data6_phishing], ignore_index=True)
    data6['label'] = (data6["label"] == 0).astype(int)
    data6 = data6[['url', 'label']]

    # Đọc dataset 7
    data7 = pd.read_csv('src/dataset/data/data7/data_bal - 20000.csv')
    data7 = data7.rename(columns={"Labels": 'label', 'URLs': 'url'})
    data7['label'] = (data7["label"] == 0).astype(int)

    # Đọc dataset 8
    data8_test = pd.read_csv('src/dataset/data/data8/test.csv')
    data8_test["label"] = (data8_test["label"] == 'legitimate').astype(int)

    # Tạo data_train
    data_train = pd.concat([data2, data3, data4, data6, data8_test], ignore_index=True)
    data_train = data_train.dropna(subset=["url", "label"])
    data_train = data_train[data_train["url"].apply(lambda x: isinstance(x, str) and x.isascii())]
    data_train = data_train.drop_duplicates()

    # Tạo data_test
    data_test = data7
    data_test = data_test[data_test["url"].apply(lambda x: x.isascii())]
    data_test = data_test.drop_duplicates()
    print
    return [data_train,data_test]

if __name__ == "__main__":
    data_train,data_test= merge_dataset()
    # Lưu dữ liệu đã xử lý
    data_train.to_csv("src/data_processing/raw/data_train_raw.csv", index=False)
    data_test.to_csv("src/data_processing/raw/data_test_raw.csv", index=False)
