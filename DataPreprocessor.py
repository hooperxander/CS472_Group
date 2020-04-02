#import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def get_train_test_data(self):
        data = pd.read_csv('data.csv')
        data_list = data.values
        #data_list = torch.tensor(data.values).tolist()

        X = []
        y = []
        for i in range(len(data_list)):
            row = []
            for j in range(len(data_list[i])):
                if j == len(data_list[i]) - 1:
                    y.append(data_list[i][j])
                else:                    row.append(data_list[i][j])
            X.append(row)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75)
        return X_train, X_test, y_train, y_test
