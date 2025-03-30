import pandas as pd

def delete_duplicate(data_train, data_test):
    # data_train = pd.read_csv('dataset/feature/data_train.csv')
    # data_test=pd.read_csv('dataset/feature/data_test.csv')
    print('----------data train---------')
    print(data_train)

    print('-------delete duplicate data----------------')
    data_train=data_train.drop_duplicates()
    print(data_train)

    data_train.to_csv('dataset/feature/data_train.csv',index=None)

    print('-----------data test-------------')
    print(data_test)
    print('-----------------data duplicate test----------')
    data_test=data_test.drop_duplicates()
    print(data_test)
    data_test.to_csv('dataset/feature/data_test.csv',index=None)