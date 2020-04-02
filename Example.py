import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from DataPreprocessor import DataPreprocessor


class BaseLine:
    def fit(self):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        x = np.array(X)
        error = y - x[:, 0]
        return np.sum(error ** 2)

    def get_params(self, deep=False):
        return []

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_data()


    clf = MLPRegressor(solver='adam', hidden_layer_sizes=(240, 240))
    clf.fit(X_train, y_train)
    # print(clf.predict(X_test))
    score = clf.score(X_test,y_test)
    print(score)

    knn = KNeighborsRegressor(n_neighbors=50)
    knn.fit(X_train, y_train)
    k_score = knn.score(X_test,y_test)
    print(k_score)
    # print(knn.predict(X_test))

    reg = LinearRegression().fit(X_train, y_train)
    r_score = reg.score(X_test,y_test)
    print(r_score)


    # BASELINE
    # total = 0
    # for index in range(10):
    #     X_train, X_test, y_train, y_test = preprocessor.get_train_test_data()
    #     x = np.array(X_test)
    #     error = y_test - x[:, 0]
    #     total = total + np.sum(error ** 2)
    # print("average SSE" + str(total/10))

