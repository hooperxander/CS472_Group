from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

class FeatureSelector:
    def select_features(self, X_train, y_train, X_test, k_features=30):
        sfs = SFS(LinearRegression(),
                   k_features=k_features,
                   forward=True,
                   floating=False,
                   scoring = 'r2',
                   cv = 0)
        sfs.fit(np.array(X_train), np.array(y_train))
        print(sfs.k_feature_names_)

        new_X_train = []
        for i in range(len(X_train)):
          new_row = []
          for j in range(len(X_train[i])):
            j_str = str(j)
            if j_str in sfs.k_feature_names_:
              new_row.append(X_train[i][j])
          new_X_train.append(new_row)

        new_X_test = []
        for i in range(len(X_test)):
          new_row = []
          for j in range(len(X_test[i])):
            j_str = str(j)
            if j_str in sfs.k_feature_names_:
              new_row.append(X_test[i][j])
          new_X_test.append(new_row)

        return new_X_train, new_X_test
