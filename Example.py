import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from CS472_Group.DataPreprocessor import DataPreprocessor
from CS472_Group.FeatureSelector import FeatureSelector

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

preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.get_train_test_data()

# Feature selection
selector = FeatureSelector()
new_X_train, new_X_test = selector.select_features(X_train, y_train, X_test, 30)

clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(10, 10, 10), random_state=1)
clf.fit(X_train, y_train)
print(clf.predict(X_test))

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

print(knn.predict(X_test))

# BASELINE
total = 0
for index in range(10):
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_data()
    x = np.array(X_test)
    error = y_test - x[:, 0]
    total = total + np.sum(error ** 2)
print("average SSE" + str(total/10))

