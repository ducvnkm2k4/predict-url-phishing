import pandas as pd
import os
def merge_dataset():
    #data 1
    data1=pd.read_csv("dataset/data1/malicious_phish.csv")
    data1["label"]=1
    data1.loc[data1['type']!="benign","label"] = 0
    del data1["type"]
    #data 2
    data2=pd.read_csv("dataset/data2/dataset_phishing.csv")
    data2=pd.DataFrame(data2,columns=["url","status"])
    data2["label"]=1
    data2.loc[data2["status"]!="legitimate","label"] = 0
    del data2["status"]
    #data3
    data3_feature = pd.read_csv("dataset/data3/features.csv")
    data3_label = pd.read_csv("dataset/data3/targets.csv")
    data3 = pd.DataFrame()
    data3['url'] = data3_feature['URL']
    # Gộp hai DataFrame lại (theo chỉ số hàng)
    data3 = pd.concat([data3, data3_label], axis=1)

    # Hiển thị kết quả
    print(data3)
    #merge data
    data13=pd.concat([data1,data3],ignore_index=True)
    
    os.makedirs('dataset/raw',exist_ok=True)
    data2.to_csv("dataset/raw/data_test_raw.csv",index=False)
    data13.to_csv("dataset/raw/data_train_raw.csv",index=False)

merge_dataset()