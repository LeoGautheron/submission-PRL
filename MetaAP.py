import numpy as np
from mytree import DecisionTreeClassifier as mytreeclf


def AP(neg1, pos1, neg2=0, pos2=0):
    return (pos1*pos1)/((pos1+pos2)*(pos1+neg1)) + pos2/(pos1+pos2+neg1+neg2)


class MetaAP():
    def __init__(self, max_depth, bestresponse):
        self.max_depth = max_depth
        self.bestresponse = bestresponse

    def learnInternalTree(self, X, y):  # return tree and list of left leaves
        dt = mytreeclf(max_depth=self.max_depth, criterion="AP",
                       random_state=1)
        dt.fit(X, y)
        leaves = np.logical_and(dt.tree_.children_left == -1,
                                dt.tree_.children_right == -1)
        if sum(leaves) < 2:
            return (dt, [0])
        samplesLeaves = np.squeeze(dt.tree_.value[leaves], axis=1)
        bestClassIdx = np.where(dt.classes_ == self.bestresponse)[0][0]
        otherClassIdx = np.where(dt.classes_ != self.bestresponse)[0][0]
        pcInit = sum(y == self.bestresponse)
        posList = samplesLeaves[:, bestClassIdx]
        negList = samplesLeaves[:, otherClassIdx]
        leftList = posList + negList
        recallList = posList / pcInit
        precisionListInv = posList / leftList
        precisionListInv = 1 - precisionListInv
        with np.errstate(divide='ignore'):  # rm warning. Divide by 0->infinity
            crVecpr = precisionListInv / recallList
        idx = np.argsort(crVecpr)
        posListOrd = posList[idx]
        negListOrd = negList[idx]
        posListRev = np.cumsum(posListOrd[::-1])[:-1]
        posListRev = np.concatenate((posListRev[::-1], [0]))
        negListRev = np.cumsum(negListOrd[::-1])[:-1]
        negListRev = np.concatenate((negListRev[::-1], [0]))
        idxsLeaves = np.where(leaves == 1)[0]  # 1 <=> True
        temp = list(map(AP, np.cumsum(negListOrd), np.cumsum(posListOrd),
                    negListRev, posListRev))
        out = idxsLeaves[idx[:np.argmax(temp)+1]]
        return (dt, out)

    def predictInternalTree(self, tree, X):
        preds = tree[0].apply(X)  # predict the leaf index
        for i, pred in enumerate(preds):
            if pred in tree[1]:  # if among the left leaves, set pred to -1
                preds[i] = -1
            else:  # else, among right leaves, set pred to 1
                preds[i] = 1
        return preds

    def fit(self, X, y):
        assert len(np.unique(y)) == 2  # The code requires two classes
        nbNode = 0
        curscore = 1
        stack = [0]
        depth = {0: 1}
        idxs = {0: np.arange(X.shape[0])}
        nodeOrder = {}
        self.trees = {}
        self.kids = {}
        while len(stack) > 0:
            idNode = stack.pop()  # gets last element
            nodeOrder[idNode] = curscore
            curscore += 1
            if depth[idNode] > self.max_depth:
                continue
            idx = idxs[idNode]  # indices of examples in the current node
            ynode = y[idx]
            if len(np.unique(ynode)) < 2:  # check purity
                continue
            Xnode = X[idx]
            tree = self.learnInternalTree(Xnode, ynode)
            preds = self.predictInternalTree(tree, Xnode)
            left = preds <= 0
            right = preds > 0
            if sum(left) == 0 or sum(right) == 0:  # check purity
                continue
            self.trees[idNode] = tree
            self.kids[idNode] = (nbNode+1, nbNode+2)
            depth[nbNode+1], depth[nbNode+2] = depth[idNode]+1, depth[idNode]+1
            idxs[nbNode+1], idxs[nbNode+2] = idx[left], idx[right]
            stack.extend([nbNode+2, nbNode+1])  # depth first: left added last
            nbNode += 2
        nbNode += 1
        self.score = np.zeros(nbNode)
        leafs = []
        for i in range(nbNode):
            if i not in self.trees:
                leafs.append((nodeOrder[i], i))
        leafs = list(sorted(leafs))
        nbLeaf = len(leafs)
        for i in range(nbLeaf):
            self.score[leafs[i][1]] = (nbLeaf - i) / nbLeaf

    def getDistances(self, tree, X):
        dists = np.ones(X.shape[0])*1e10
        features = tree[0].tree_.feature
        threshold = tree[0].tree_.threshold
        children_left = tree[0].tree_.children_left
        children_right = tree[0].tree_.children_right
        value = []
        for v in tree[0].tree_.value:
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
            dists /= maxi*100
        return dists

    def predict(self, X):
        retId = np.array([0] * X.shape[0])  # 0 is root
        idxs = {0: np.arange(X.shape[0])}
        stack = [(0, self.trees[0])]
        dists = np.array([0] * X.shape[0], dtype=np.float64)
        while len(stack) > 0:
            idNode, tree = stack.pop()
            idx = idxs[idNode]  # indices of examples in the current node
            if idx.shape[0] == 0:
                continue
            if idNode not in self.trees:
                retId[idx] = idNode
                dists[idx] = self.getDistances(tree, X[idx])
                continue
            preds = self.predictInternalTree(self.trees[idNode], X[idx])
            kids = self.kids[idNode]
            idxs[kids[0]], idxs[kids[1]] = idx[preds <= 0], idx[preds > 0]
            stack.extend([(kids[1], self.trees[idNode]),
                          (kids[0], self.trees[idNode])])
        return self.score[retId]+dists


class MetaAPForest():
    def __init__(self, bestresponse, n_estimators=100, max_depth=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.bestresponse = bestresponse

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            preds += np.array(self.estimators[i].predict(X))
        return preds / self.n_estimators

    def fit(self, X, y):
        self.estimators = []
        n_samples = X.shape[0]
        for i in range(self.n_estimators):
            indices = self.random_state.randint(0, n_samples, n_samples)
            clf = MetaAP(max_depth=self.max_depth,
                         bestresponse=self.bestresponse)
            clf.fit(X[indices], y[indices])
            self.estimators.append(clf)
