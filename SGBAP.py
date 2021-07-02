import numpy as np
from sklearn.tree import DecisionTreeRegressor


def negative_gradient(y, pred):
    pred = pred.ravel()
    pred = 1/(1+np.exp(-pred))
    dSig = pred * (1 - pred)
    Sum = np.sum(pred)
    df = (-1*(y == 1)*dSig*np.sum(pred[y != 1]) +
          ((y != 1)*dSig*Sum - np.sum(pred[y != 1])*dSig))/Sum
    return -np.array(df)


class SGBAP:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        self.estimators = [DecisionTreeRegressor(
                                           max_depth=self.max_depth).fit(X, y)]
        y_pred = self.estimators[0].predict(X)
        for i in range(1, self.n_estimators):
            residual = negative_gradient(y, y_pred)
            self.estimators.append(DecisionTreeRegressor(
                                    max_depth=self.max_depth).fit(X, residual))
            y_pred += self.learning_rate * self.estimators[i].predict(X)
        return self

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0]))
        for i in range(self.n_estimators):
            predictions += self.estimators[i].predict(X)
        predictions = predictions/float(self.n_estimators)
        probas = 1/(1+np.exp(-predictions))
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        probas[probas > 0.5] = 1
        probas[probas <= 0.5] = 0
        return probas
