import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth

    def _validate_y(self, y):
        y = np.copy(y)
        y_store_unique_indices = np.zeros(y.shape, dtype=int)
        classes, y_store_unique_indices = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = classes.shape[0]
        y = y_store_unique_indices
        return y

    def getDistances(self, tree, X):
        dists = np.ones(X.shape[0])*1e10
        features = tree.feature
        threshold = tree.threshold
        children_left = tree.children_left
        children_right = tree.children_right
        value = []
        for v in tree.value:
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
                if dist < dists[i]:
                    dists[i] = dist
                if X[i, features[node]] <= threshold[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            dists[i] *= value[node]
        dists -= np.min(dists)
        maxi = np.max(dists)
        if maxi != 0:
            dists /= maxi
        return dists

    def predict(self, X):
        all_proba = np.zeros(X.shape[0], dtype=np.float64)
        for e in self.estimators_:
            all_proba += self.getDistances(e.tree_, X)
        return all_proba / self.n_estimators

    def fit(self, X, y):
        y = self._validate_y(y)
        self.estimators_ = []
        n_samples = X.shape[0]
        for i in range(self.n_estimators):
            rnd = np.random.RandomState(self.random_state.randint(
                                                       np.iinfo(np.int32).max))
            tree = DecisionTreeClassifier(
                      criterion=self.criterion, max_depth=self.max_depth,
                      min_samples_split=2, min_samples_leaf=1,
                      min_weight_fraction_leaf=0, max_features=None,
                      ccp_alpha=0.0, max_leaf_nodes=None, random_state=rnd,
                      min_impurity_decrease=0, min_impurity_split=None)
            indices = rnd.randint(0, n_samples, n_samples)
            sample_counts = np.bincount(indices, minlength=n_samples)
            tree.fit(X, y, sample_weight=sample_counts)
            self.estimators_.append(tree)
