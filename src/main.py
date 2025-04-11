import kagglehub
import shutil
import os
import pandas as pd  
from sympy import im
from ucimlrepo import fetch_ucirepo 
from data_processing.merge_datasets import merge_dataset
from data_processing.merge_datasets import merge_dataset
from data_processing.char_probabilities import char_pro
from data_processing.feature_engineering import parallel_feature_extraction
from data_processing.pre_processing.data_normalization import data_normalization
from trainning.random_forest import train_random_forest
from trainning.decision_tree import train_decision_tree
from trainning.xgboot import train_xgboost
from trainning.support_vector_machine import train_svm
from trainning.KNN import train_knn
from trainning.logistic_regression import train_logistic_regression

import pandas as pd  

import wget
import sys
import time
sys.dont_write_bytecode = True
import sys
import time
sys.dont_write_bytecode = True
def download_and_move(url, save_path):
    """Tải dataset từ KaggleHub và lưu vào đúng thư mục"""
    downloaded_path = kagglehub.dataset_download(url)

    # Tạo thư mục lưu nếu chưa có
    os.makedirs(save_path, exist_ok=True)

    # Kiểm tra nếu dữ liệu tải về là file duy nhất
    if os.path.isfile(downloaded_path):
        shutil.move(downloaded_path, os.path.join(save_path, os.path.basename(downloaded_path)))
    else:
        for file_name in os.listdir(downloaded_path):
            shutil.move(os.path.join(downloaded_path, file_name), save_path)
        shutil.rmtree(downloaded_path)

def download_dataset():
    # 📌 Tải dataset từ KaggleHub
    # url_data1 = 'sid321axn/malicious-urls-dataset'
    # save_path_data1 = 'dataset/data/data1'
    # download_and_move(url_data1, save_path_data1)

    url_data2 = "shashwatwork/web-page-phishing-detection-dataset"
    save_path_data2 = 'src/dataset/data/data2'
    save_path_data2 = 'src/dataset/data/data2'
    download_and_move(url_data2, save_path_data2)

    # 📌 Tải dataset từ UCI và lưu vào dataset/data3/
    save_path_data3 = 'src/dataset/data/data3'
    save_path_data3 = 'src/dataset/data/data3'
    os.makedirs(save_path_data3, exist_ok=True)

    phiusiil_phishing_url_website = fetch_ucirepo(id=967)  
    # 📌 Lưu dataset dưới dạng CSV
    X = phiusiil_phishing_url_website.data.features
    y = phiusiil_phishing_url_website.data.targets

    X.to_csv(os.path.join(save_path_data3, "features.csv"), index=False)
    y.to_csv(os.path.join(save_path_data3, "targets.csv"), index=False)
    
    # top 100k tranco
    url='https://tranco-list.eu/download/5897N/100000'
    top_100k_save_path="src/dataset/tranco_list"
    wget.download(url,top_100k_save_path)

def train_model(data_train,data_test,data_train_scaled,data_test_scaled):
    train_decision_tree(data_train,data_test,True)
    print('-------------------time sleep-----------------')
    time.sleep(1200)

    train_xgboost(data_train,data_test,True)
    print('-------------------time sleep-----------------')
    time.sleep(1200)

    train_random_forest(data_train,data_test,True)
    print('-------------------time sleep-----------------')
    time.sleep(1200)
    
    # train_svm(data_train_scaled,data_test_scaled,True)
    # print('-------------------time sleep-----------------')
    # time.sleep(1200)

    # train_knn(data_train_scaled,data_test_scaled)
    # print('-------------------time sleep-----------------')
    # time.sleep(1200)
    
    # train_logistic_regression(data_train_scaled,data_test_scaled,True)

if __name__ == "__main__":
    os.makedirs('src/data_processing/feature',exist_ok=True)
    os.makedirs('src/data_processing/raw',exist_ok=True)
    os.makedirs('src/model/model',exist_ok=True)
    os.makedirs('src/model/report',exist_ok=True)
    # download_dataset()
    data_train,data_test= merge_dataset()
    char_probabilities = char_pro().set_index('Character')['Probability'].to_dict()
    # Đọc danh sách top 100k domain từ Tranco
    top_100k_tranco_list = set(pd.read_csv('src/dataset/tranco_list/tranco_5897N.csv', header=None).iloc[:, 1].tolist())

    extracted_features_train = parallel_feature_extraction(data_train['url'], data_train['label'], char_probabilities, top_100k_tranco_list)
    extracted_features_test = parallel_feature_extraction(data_test['url'], data_test['label'], char_probabilities, top_100k_tranco_list)


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

    data_train_feature=data_train.drop_duplicates()
    data_test_feature=data_test.drop_duplicates()

    data_train_scaled,data_test_scaled = data_normalization(data_train_feature,data_test_feature)
    train_model(data_train_feature,data_test_feature,data_train_scaled,data_test_scaled)

    data_train_feature.to_csv('src/data_processing/feature/data_train.csv',index=None)
    data_test_feature.to_csv('src/data_processing/feature/data_test.csv',index=None)
    data_train.to_csv("src/data_processing/raw/data_train_raw.csv", index=False)
    data_test.to_csv("src/data_processing/raw/data_test_raw.csv", index=False)
    data_train_scaled.to_csv("src/data_processing/feature/data_train_scaled.csv", index=False)
    data_test_scaled.to_csv("src/data_processing/feature/data_test_scaled.csv", index=False)
