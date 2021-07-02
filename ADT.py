import numpy as np
from mytree import DecisionTreeClassifier as mytreeclf


class ADT():
    def __init__(self, criterion, max_depth, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.clf = mytreeclf(max_depth=self.max_depth,
                             criterion=self.criterion,
                             random_state=self.random_state)
        self.clf.fit(X, y)

    def predict(self, X):
        preds = np.ones(X.shape[0])*1e10
        features = self.clf.tree_.feature
        threshold = self.clf.tree_.threshold
        children_left = self.clf.tree_.children_left
        children_right = self.clf.tree_.children_right
        value = []
        for v in self.clf.tree_.value:
            if v[0][0] < v[0][1]:
                value.append(1)
            elif v[0][0] > v[0][1]:
                value.append(-1)
            else:
                value.append(0)
        for i in range(X.shape[0]):
            node = 0
            while children_left[node] > 0:
                dist = abs(X[i, features[node]]-threshold[node])
                if dist < preds[i]:
                    preds[i] = dist
                if X[i, features[node]] <= threshold[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            preds[i] *= value[node]
        preds -= np.min(preds)
        maxi = np.max(preds)
        if maxi != 0:
            preds /= maxi
        return preds
