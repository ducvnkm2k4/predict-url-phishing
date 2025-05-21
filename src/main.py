import os
import pandas as pd  
from data_processing.merge_datasets import merge_dataset
from data_processing.char_probabilities import char_pro
from data_processing.feature_engineering import parallel_feature_extraction
from train_model.random_forest import train_random_forest
from train_model.decision_tree import train_decision_tree
from train_model.train_xgboost import train_xgboost
from train_model.support_vector_machine import train_svm
from train_model.KNN import train_knn
from train_model.logistic_regression import train_logistic_regression
import sys

sys.dont_write_bytecode = True

def train_model(data_train,data_test):
    train_decision_tree(data_train,data_test)
    train_xgboost(data_train,data_test)
    train_random_forest(data_train,data_test)
    train_knn(data_train,data_test)
    train_logistic_regression(data_train,data_test)
    train_svm(data_train,data_test)

if __name__ == "__main__":
    data_train,data_test= merge_dataset()
    print('---------complete merge dataset------------')
    char_probabilities = char_pro().set_index('Character')['Probability'].to_dict()
    print('--------------complete calculate char probilities-------')
    # Đọc danh sách top 100k domain từ Tranco
    top_100k_tranco_list = set(pd.read_csv('src/dataset/tranco_list/tranco_5897N.csv', header=None).iloc[:, 1].tolist())

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

    data_train_feature=data_train_feature.drop_duplicates().dropna()
    data_test_feature=data_test_feature.drop_duplicates().dropna()
    print('------------------complete extract feature--------------------')
    train_model(data_train_feature,data_test_feature)
    print('--------------------complete----------------------')
    data_train_feature.to_csv('src/output/data/data_train.csv', index=False)
    data_test_feature.to_csv('src/output/data/data_test.csv', index=False)
    data_train.to_csv('src/output/data/data_train_raw.csv', index=False)
    data_test.to_csv('src/output/data/data_test_raw.csv', index=False)