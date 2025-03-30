import pandas as pd
import os
def merge_dataset():
    os.makedirs("data_processing/raw", exist_ok=True)
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

    #data5
    data5_raw=pd.read_csv('dataset/data/data5/combined_dataset.csv')
    data5=pd.DataFrame()
    data5["url"]=data5_raw["domain"].str.strip("'\"")
    data5["label"]=(data5_raw["label"]==0).astype(int)

    #data 6
    # Đọc dữ liệu
    data6_legimate = pd.read_csv('dataset/data/data6/extracted_legitmate_dataset.csv')
    data6_phishing = pd.read_csv('dataset/data/data6/extracted_phishing_dataset.csv')
    data6_legimate['url'] = data6_legimate['protocol'] + '://' + data6_legimate['domain_name'] + '/' + data6_legimate["address"]
    data6_phishing['url'] = data6_phishing['protocol'] + '://' + data6_phishing['domain_name'] + '/' + data6_phishing["address"]
    data6 = pd.concat([data6_legimate, data6_phishing], ignore_index=True)
    data6['label'] = (data6["label"] == 0).astype(int)
    data6 = data6[['url', 'label']]

    #data 7
    data7 = pd.read_csv('dataset/data/data7/data_bal - 20000.csv')
    data7=data7.rename(columns={"Labels":'label', 'URLs':'url'})
    data7['label'] = (data7["label"]==0).astype(int)
    #data 8
    data8_test=pd.read_csv('dataset/data/data8/test.csv')
    data8_train=pd.read_csv('dataset/data/data8/train.csv')
    data8_val=pd.read_csv('dataset/data/data8/val.csv')
    data8_test["label"]=(data8_test["label"]=='legitimate').astype(int)
    data8_train["label"]=(data8_train["label"]=='legitimate').astype(int)
    data8_val["label"]=(data8_val["label"]=="legitimate").astype(int)

    # create data train test
    data_train = pd.concat([data1,data3,data4,data6,data7,data8_train,data8_val], ignore_index=True)
    data_train = data_train.dropna(subset=["url"])
    data_train = data_train[data_train["url"].apply(lambda x: isinstance(x, str) and x.isascii())]

    print("-------------data train------------------")
    print(data_train)
    data_train= data_train.drop_duplicates()
    print("-------------data train delete------------------")
    print(data_train)
    data_train.to_csv("data_processing/raw/data_train_raw.csv", index=False)

    #data test
    data2 = data2[data2["url"].apply(lambda x: x.isascii())]
    data_test = pd.concat([data2,data5,data8_test],ignore_index=True)
    data_test = data_test[data_test["url"].apply(lambda x: x.isascii())]
    print("---------------data test--------------")
    print(data_test)
    data_test=data_test.drop_duplicates()
    print("---------------data test delete--------------")
    print(data_test)
    data_test.to_csv("data_processing/raw/data_test_raw.csv", index=False)
    
   

merge_dataset()
