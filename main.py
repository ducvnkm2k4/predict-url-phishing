import kagglehub
import shutil
import os
import pandas as pd  
from ucimlrepo import fetch_ucirepo 
from data_processing.merge_datasets import merge_dataset
from data_processing.char_probabilities import char_pro
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
    url_data1 = 'sid321axn/malicious-urls-dataset'
    save_path_data1 = 'dataset/data/data1'
    download_and_move(url_data1, save_path_data1)

    url_data2 = "shashwatwork/web-page-phishing-detection-dataset"
    save_path_data2 = 'dataset/data/data2'
    download_and_move(url_data2, save_path_data2)

    # 📌 Tải dataset từ UCI và lưu vào dataset/data3/
    save_path_data3 = 'dataset/data/data3'
    os.makedirs(save_path_data3, exist_ok=True)

    phiusiil_phishing_url_website = fetch_ucirepo(id=967)  
    # 📌 Lưu dataset dưới dạng CSV
    X = phiusiil_phishing_url_website.data.features
    y = phiusiil_phishing_url_website.data.targets

    X.to_csv(os.path.join(save_path_data3, "features.csv"), index=False)
    y.to_csv(os.path.join(save_path_data3, "targets.csv"), index=False)

    #data5
    url='kunal4892/phishingandlegitimateurls'
    save_path_data5='dataset/data/data5'
    download_and_move(url,save_path_data5)
    
    # top 100k tranco
    url='https://tranco-list.eu/download/5897N/100000'
    top_100k_save_path="dataset/tranco_list"
    wget.download(url,top_100k_save_path)

def train_model():
    # Load dữ liệu
    data_train = pd.read_csv('data_processing/feature/data_train.csv')
    data_test = pd.read_csv('data_processing/feature/data_test.csv')
    data_train_scaled = pd.read_csv('data_processing/feature/data_train_scaled.csv')
    data_test_scaled = pd.read_csv('data_processing/feature/data_test_scaled.csv')
    train_decision_tree(data_train,data_test,True)
    print('-------------------time sleep-----------------')
    time.sleep(1200)

    train_xgboost(data_train,data_test,True)
    print('-------------------time sleep-----------------')
    time.sleep(1200)

    train_random_forest(data_train,data_test,True)
    print('-------------------time sleep-----------------')
    time.sleep(1200)
    
    train_svm(data_train_scaled,data_test_scaled,True)
    print('-------------------time sleep-----------------')
    time.sleep(1200)

    train_knn(data_train_scaled,data_test_scaled)
    print('-------------------time sleep-----------------')
    time.sleep(1200)
    
    train_logistic_regression(data_train_scaled,data_test_scaled,True)


def main():
    # download_dataset()
    # merge_dataset()
    # char_pro()
    train_model()

if __name__ == "__main__":
    main()
