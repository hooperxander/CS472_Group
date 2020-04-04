from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
import numpy as np

class FeatureSelector:
    def select_features(self, model, X_train, y_train, k_features=(1,30), scorer='r2', cv=0):
        sfs = SFS(model,
                   k_features=k_features,
                   forward=True,
                   floating=False,
                   scoring = scorer,
                   cv = cv,
                   verbose=2)
        sfs.fit(np.array(X_train), np.array(y_train))
        print(sfs.k_feature_idx_)

        """
        new_X_train = []
        for i in range(len(X_train)):
          new_row = []
          for j in range(len(X_train[i])):
            j_str = str(j)
            if j_str in sfs.k_feature_names_:
              new_row.append(X_train[i][j])
          new_X_train.append(new_row)


        return new_X_train
        """