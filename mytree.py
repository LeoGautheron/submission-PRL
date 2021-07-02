import numpy as np
# Compile .pyx files (using Cython) to be able to import them just after
import pyximport
pyximport.install(reload_support=True, language_level=2,
                  setup_args={'include_dirs': np.get_include()})
# Import the Cython files which are automatically compiled by pyximport
import _tree
DOUBLE = _tree.DOUBLE
CRITERIA_CLF = {"gini": _tree.Gini, "entropy": _tree.Entropy,
                "AP": _tree.AveragePrecision}
# Using criterion="AP", the algorithm supposes that there are only two unique
# values in the y parameter of the fit function


class DecisionTreeClassifier():  # based on sklearn code
    def __init__(self, criterion, max_depth, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]
        y = np.copy(y)
        self.classes_ = []
        self.n_classes_ = []
        y_encoded = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                   return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_encoded
        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_fraction_leaf = 0
        min_weight_leaf = (min_weight_fraction_leaf * n_samples)
        min_impurity_split = 0
        min_impurity_decrease = 0
        criterion = self.criterion
        if not isinstance(criterion, _tree.Criterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                     self.n_classes_)
        splitter = _tree.Splitter(criterion, n_features,
                                  min_samples_leaf, min_weight_leaf,
                                  np.random.RandomState(self.random_state))
        self.tree_ = _tree.Tree(n_features, self.n_classes_, self.n_outputs_)
        builder = _tree.TreeBuilder(splitter, min_samples_split,
                                    min_samples_leaf, min_weight_leaf,
                                    max_depth, min_impurity_decrease,
                                    min_impurity_split)
        builder.build(self.tree_, X, y)
        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
        return self

    def predict(self, X):
        proba = self.tree_.predict(X.astype(np.float32))
        n_samples = X.shape[0]
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)
        else:
            class_type = self.classes_[0].dtype
            predictions = np.zeros((n_samples, self.n_outputs_),
                                   dtype=class_type)
            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                                        np.argmax(proba[:, k], axis=1), axis=0)
            return predictions

    def predict_proba(self, X):
        proba = self.tree_.predict(X.astype(np.float32))
        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer
            return proba
        else:
            all_proba = []
            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)
            return all_proba

    def apply(self, X):
        return self.tree_.apply(X.astype(np.float32))
