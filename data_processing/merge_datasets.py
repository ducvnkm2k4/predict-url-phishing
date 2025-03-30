import pandas as pd
import os

def merge_dataset():
    # Đọc dataset 1
    data1 = pd.read_csv("dataset/data/data1/malicious_phish.csv")
    data1["label"] = (data1["type"] == "benign").astype(int)  # Chuyển đổi nhãn
    data1.drop(columns=["type"], inplace=True)  # Xóa cột type
    data1['url']=data1["url"].str.strip("'\"")
    # Đọc dataset 2
    data2 = pd.read_csv("dataset/data/data2/dataset_phishing.csv")[["url", "status"]]
    data2["label"] = (data2["status"] == "legitimate").astype(int)
    data2.drop(columns=["status"], inplace=True)

    # Đọc dataset 3
    data3_feature = pd.read_csv("dataset/data/data3/features.csv")
    data3_label = pd.read_csv("dataset/data/data3/targets.csv")
    data3 = pd.DataFrame({"url": data3_feature["URL"], "label": data3_label["label"]})

    # dataset 4
    data4_raw=pd.read_csv('dataset/data/data4/urlset.csv',encoding='latin1')
    data4=pd.DataFrame()

    data4['url']=data4_raw['domain'].str.strip("'\"")
    data4["label"]= (data4_raw["label"]==0.0).astype(int)
    # Gộp data1 và data3 thành tập train
    data_train = pd.concat([data1, data3,data4], ignore_index=True)

    # Lọc bỏ URL chứa ký tự không phải ASCII (tránh lỗi encoding)
    data_train = data_train[data_train["url"].apply(lambda x: x.isascii())]
    data2 = data2[data2["url"].apply(lambda x: x.isascii())]

    print(data_train)
    # Tạo thư mục lưu dataset nếu chưa có
    os.makedirs("dataset/raw", exist_ok=True)

    # Lưu dataset đã làm sạch
    data_train.to_csv("dataset/raw/data_train_raw.csv", index=False)
    data2.to_csv("dataset/raw/data_test_raw.csv", index=False)

    

merge_dataset()
