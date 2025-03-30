import pandas as pd
import os
def delete_duplicate(data_train, data_test):
    os.makedirs('ata_processing/feature',exist_ok=True)
    print('----------data train---------')
    print(data_train)

    print('-------delete duplicate data----------------')
    data_train=data_train.drop_duplicates()
    print(data_train)

    data_train.to_csv('data_processing/feature/data_train.csv',index=None)

    print('-----------data test-------------')
    print(data_test)
    print('-----------------data duplicate test----------')
    data_test=data_test.drop_duplicates()
    print(data_test)
    data_test.to_csv('data_processing/feature/data_test.csv',index=None)