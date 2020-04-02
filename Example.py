import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from DataPreprocessor import DataPreprocessor
from scipy.stats import uniform


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
X_train, X_test, y_train, y_test = preprocessor.get_train_test_data(norm=True)

#BClass = MLPRegressor(max_iter=10000)
parameter_space = {
    'hidden_layer_sizes': [(1000,), (300, 300), (100, 100, 100, 100)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.01, 0.001, 0.0001, 0.00001, 0.000001],
    'learning_rate': ['constant', 'adaptive'],
    'momentum': uniform(loc=0.1, scale=0.8),
    'early_stopping': [True, False]
}

#clf = RandomizedSearchCV(BClass, parameter_space, n_jobs=6, cv=3, verbose=10, n_iter=50)
#clf.fit(X_train, y_train)
#print('Best parameters found:\n', clf.best_params_)

#Best settings found by random search:
#BClass = MLPRegressor(max_iter=10000, activation='logistic', alpha=0.001, early_stopping=False,
#                      hidden_layer_sizes=[300, 300], learning_rate='constant', momentum=0.48757434953041645, solver='adam')
#BClass.fit(X_train, y_train)

BClass = MLPRegressor(max_iter=10000, hidden_layer_sizes=[1000])
BClass.fit(X_train, y_train)
pred = BClass.predict(X_test)
error = y_test - pred
total = np.sum(error ** 2)
print(total)

#knn = KNeighborsRegressor(n_neighbors=3)
#knn.fit(X_train, y_train)

#print(knn.predict(X_test))

# BASELINE
total = 0
for index in range(10):
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_data()
    x = np.array(X_test)
    error = y_test - x[:, 0]
    total = total + np.sum(error ** 2)
print("average SSE " + str(total/10))

