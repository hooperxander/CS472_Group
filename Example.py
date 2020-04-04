import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from DataPreprocessor import DataPreprocessor
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from scipy.stats import uniform
from FeatureSelector import *
from sklearn.utils import shuffle


preprocessor = DataPreprocessor()

def mse_error(y_true, y_pred):
    error = y_true - y_pred
    return np.sum(error ** 2)/(len(y_pred))


X_train, y_train = preprocessor.get_train_test_data(norm=True, test=0)
BClass = MLPRegressor(max_iter=10000, hidden_layer_sizes=[50,50,50,50], early_stopping=True, validation_fraction=.1, n_iter_no_change=300,
                      activation='tanh', alpha=0.00001, learning_rate='adaptive', momentum=0.3, solver='sgd')

scorer = make_scorer(mse_error, greater_is_better=False)

"""
#Search hyper parameter tuning stuff

parameter_space = {
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.0001, 0.00001],
    'learning_rate': ['constant', 'adaptive'],
    'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
}

clf = GridSearchCV(BClass, parameter_space, n_jobs=6, cv=3, verbose=10, scoring=scorer)
clf.fit(X_train, y_train)
print('Best parameters found:\n', clf.best_params_)
"""


"""
#Feature selection stuff

fs = FeatureSelector()
fs.select_features(BClass, X_train, y_train, scorer=scorer, k_features=(1, 4))
"""

#Cross validation test. Averaging 5 different 6-fold CV averages
total = 0
iters = 5
cross=6
for i in range(iters):
    shuffle(X_train, y_train)
    scores = cross_validate(BClass, X_train, y_train, scoring=scorer, cv=cross, n_jobs=6, return_train_score=True, verbose=10)
    print(scores['test_score'])
    total += np.sum(scores['test_score']) / cross
total /= iters
print("Average: ", total)

"""
#Baseline MSE = 15.18
#This uses the whole file, so we don't ever have to re-calculate it.
# BASELINE
total = 0
#for index in range(10):
X_train, X_test, y_train, y_test = preprocessor.get_train_test_data(test=0)
x = np.array(X_train)
error = y_train - x[:, 0]
total = total + np.sum(error ** 2)
print("Baseline MSE " + str(total/x.shape[0]))
"""
