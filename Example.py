import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from CS472_Group.DataPreprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.get_train_test_data()

clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(10, 10, 10), random_state=1)
clf.fit(X_train, y_train)
print(clf.predict(X_test))

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

print(knn.predict(X_test))

# BASELINE
x = np.array(X_test)
error = y_test-x[:,0]
print("Baseline SSE: " + str(np.sum(error**2)))