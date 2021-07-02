import os
import random
from io import StringIO
import numpy as np
import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from TreeRank import TreeRank
os.utime("_tree.pxd", None)  # force recompilation of mytree used in MetaAP
os.utime("_tree.pyx", None)
import pyximport
pyximport.install(reload_support=True, language_level=2,
                  setup_args={'include_dirs': np.get_include()})
# Import the Cython files which are automatically compiled by pyximport
import _tree
from mytree import DecisionTreeClassifier as mytreeclf
from MetaAP import MetaAP
if not os.path.exists("figures"):
    try:
        os.makedirs("figures")
    except:
        pass
seed = 10
data = pd.read_csv("datasets/yeast.data", header=None, sep=r'\s+')
data = data.drop([0], axis=1)
y = np.array([1 if elt == 'EXC' else 0 for elt in data[9]])
X = np.array(data.drop([9], axis=1))
pctPos = 100*len(y[y == 1])/len(y)
np.random.seed(seed)
random.seed(seed)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, shuffle=True,
                                                stratify=y, test_size=0.3)


def export_graphviz_Entropy(clf, Xtrain, ytrain, Xtest, ytest, apTrain, apTest,
                            algo):
    out_file = StringIO()
    ranks = {'leaves': []}
    out_file.write('digraph Tree {\n')
    out_file.write('labelloc="t";\n')
    out_file.write(('label="' + algo + ', Train AP: {:5.2f}'.format(
                    apTrain) + '%, Test AP: {:5.2f}'.format(apTest) +
                    '%";\n'))
    out_file.write(('node [shape=box, style="filled", margin=0.01, ' +
                    'width=0, height=0] ;\n'))
    out_file.write(('graph [nodesep="0", ranksep="0.2", margin=0, ' +
                    'width=0, height=0];\n'))
    stack = [(0, 0, np.arange(Xtrain.shape[0]), np.arange(Xtest.shape[0]))]
    tree = clf.tree_
    while len(stack) > 0:
        node_id, depth, idxTrain, idxTest = stack.pop()
        nptrain = len(ytrain[idxTrain][ytrain[idxTrain] == 1])
        nntrain = len(idxTrain) - nptrain
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if left_child == _tree.TREE_LEAF:
            ranks['leaves'].append(str(node_id))
        elif str(depth) not in ranks:
            ranks[str(depth)] = [str(node_id)]
        else:
            ranks[str(depth)].append(str(node_id))
        out_file.write(str(node_id) + ' [label="')
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            score = nptrain/(nptrain+nntrain)
            out_file.write(('score = ' + str(round(score, 3)) + '\\n'))
        else:
            f, t = tree.feature[node_id], tree.threshold[node_id]
            out_file.write(('X' + str(f) + " <= " +
                            str(round(t, 3)) + '\\n'))
        out_file.write(('train (' + str(nntrain) + ', ' + str(nptrain) + ')"'))
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            out_file.write(', fillcolor="cyan"] ;\n')
        else:
            out_file.write(', fillcolor="white"] ;\n')

        if left_child != _tree.TREE_LEAF:
            out_file.write('%d -> %d [color="#6EAF0F"];\n' % (
                                                        node_id, left_child))
            out_file.write('%d -> %d [color="#ED5853"];\n' % (
                                                        node_id, right_child))
            lefttrain = Xtrain[idxTrain][:, f] <= t
            righttrain = np.logical_not(lefttrain)
            lefttest = Xtest[idxTest][:, f] <= t
            righttest = np.logical_not(lefttest)
            stack.append((right_child, depth + 1,
                          idxTrain[righttrain], idxTest[righttest]))
            stack.append((left_child, depth + 1,
                          idxTrain[lefttrain], idxTest[lefttest]))
    out_file.write("}")
    graph = graphviz.Source(out_file.getvalue())
    graph.render("figures/Entropy")


clf = mytreeclf(max_depth=4, criterion="entropy", random_state=1)
clf.fit(Xtrain, ytrain)
rankTrain = clf.predict_proba(Xtrain.astype(np.float32))[:, 1]
rankTest = clf.predict_proba(Xtest.astype(np.float32))[:, 1]
apTrain, apTest = (average_precision_score(ytrain, rankTrain)*100,
                   average_precision_score(ytest, rankTest)*100)
print("Entropy Train AP {:5.2f}".format(apTrain),
      "Test AP {:5.2f}".format(apTest))
export_graphviz_Entropy(
                 clf, Xtrain, ytrain, Xtest, ytest, apTrain, apTest, "Entropy")


def buildFullTree_TreeRank(clf):
    def rObjToDic(obj):
        return dict(zip(obj.names, map(list, list(obj))))
    dic = rObjToDic(clf.tree)
    scores = dic["score"]
    clf.fullTree = {}
    stack = [0]
    while len(stack) > 0:
        idNode = stack.pop()
        tree = rObjToDic(dic["LRList"][idNode])
        leftLeaves = [int(v)-1 for v in tree["Lnode"]]
        kids = [int(v)-1 for v in list(dic["kidslist"][idNode])]
        if len(list(dic["kidslist"][kids[1]])) == 2:
            stack.append(kids[1])
        if len(list(dic["kidslist"][kids[0]])) == 2:
            stack.append(kids[0])
        stackTree = [0]
        while len(stackTree) > 0:
            idTreeNode = stackTree.pop()
            curNode = str(idNode) + ";" + str(idTreeNode)
            kidsNode = [int(v)-1
                        for v in list(tree["kidslist"][idTreeNode])]
            if len(kidsNode) != 2:
                if idTreeNode in leftLeaves:
                    if len(list(dic["kidslist"][kids[0]])) != 2:
                        clf.fullTree[curNode] = {"score": scores[kids[0]]}
                    else:
                        clf.fullTree[curNode] = {"goToNode": str(kids[0])+";0"}
                else:
                    if len(list(dic["kidslist"][kids[1]])) != 2:
                        clf.fullTree[curNode] = {"score": scores[kids[1]]}
                    else:
                        clf.fullTree[curNode] = {"goToNode": str(kids[1])+";0"}
            else:
                stackTree.extend([kidsNode[1], kidsNode[0]])
                split = rObjToDic(tree["split"][idTreeNode])
                f = int(split["name"][0])
                t = float(split["breaks"][0])
                node = (f, t, str(idNode)+";"+str(kidsNode[0]),
                        str(idNode)+";"+str(kidsNode[1]))
                clf.fullTree[curNode] = {"node": node}
    nodes = []
    for curNode in clf.fullTree.keys():
        if "node" in clf.fullTree[curNode]:
            nodes.append(curNode)
    for curNode in nodes:
        f, t, k1, k2 = clf.fullTree[curNode]["node"]
        v1, v2 = k1, k2
        if "goToNode" in clf.fullTree[k1]:
            v1 = clf.fullTree[k1]["goToNode"]
            del clf.fullTree[k1]
            clf.fullTree[v1]["changeTree"] = True
        if "goToNode" in clf.fullTree[k2]:
            v2 = clf.fullTree[k2]["goToNode"]
            del clf.fullTree[k2]
            clf.fullTree[v2]["changeTree"] = True
        clf.fullTree[curNode]["node"] = (f, t, v1, v2)


clf = TreeRank(max_depth=2)
clf.fit(Xtrain, ytrain)
buildFullTree_TreeRank(clf)
rankTrain = clf.predict(Xtrain)
rankTest = clf.predict(Xtest)
apTrain, apTest = (average_precision_score(ytrain, rankTrain)*100,
                   average_precision_score(ytest, rankTest)*100)
print("TreeRank Train AP {:5.2f}".format(apTrain),
      "Test AP {:5.2f}".format(apTest))


def export_graphviz_TreeRank_MetaAP(fullTree, Xtrain, ytrain, Xtest, ytest,
                                    apTrain, apTest, algo):
    stack = [("0;0", np.arange(Xtrain.shape[0]), np.arange(Xtest.shape[0]))]
    strIdToNum = {"0;0": 0}
    lastNum = 0
    allNodes = {}
    scoreToNode = {}
    nodesOrder = {}
    while len(stack) > 0:
        idNode, idxTrain, idxTest = stack.pop()
        if idNode not in strIdToNum:
            lastNum += 1
            strIdToNum[idNode] = lastNum
        idNoden = strIdToNum[idNode]
        nptrain = len(ytrain[idxTrain][ytrain[idxTrain] == 1])
        nntrain = len(idxTrain) - nptrain
        nptest = len(ytest[idxTest][ytest[idxTest] == 1])
        nntest = len(idxTest) - nptest
        if "node" not in fullTree[idNode]:
            score = fullTree[idNode]["score"]
            if score not in scoreToNode:
                scoreToNode[score] = idNoden
            idNoden = scoreToNode[score]
            strIdToNum[idNode] = idNoden
        if idNoden not in allNodes:
            allNodes[idNoden] = {"nptrain": 0, "nntrain": 0, "nptest": 0,
                                 "nntest": 0, "childs": []}
        allNodes[idNoden]["nptrain"] += nptrain
        allNodes[idNoden]["nntrain"] += nntrain
        allNodes[idNoden]["nptest"] += nptest
        allNodes[idNoden]["nntest"] += nntest
        if "node" not in fullTree[idNode]:
            allNodes[idNoden]["score"] = fullTree[idNode]["score"]
            allNodes[idNoden]["changeTree"] = True
        else:
            (f, t, v1, v2) = fullTree[idNode]["node"]
            allNodes[idNoden]["node"] = (f, t)
            if "changeTree" in fullTree[idNode]:
                allNodes[idNoden]["changeTree"] = fullTree[idNode][
                                                                  "changeTree"]
            kids = [v1, v2]
            if v1.endswith(";0") and v2.endswith(";0") and v1 != v2:
                if int(v1.split(";")[0]) < int(v2.split(";")[0]):
                    nodesOrder[v1] = v2
                else:
                    nodesOrder[v2] = v1
            elif "node" not in fullTree[v1] and "node" not in fullTree[v2]:
                s1 = fullTree[v1]["score"]
                s2 = fullTree[v2]["score"]
                if s1 != s2:
                    if s1 < s2:
                        nodesOrder[v2] = v1
                    else:
                        nodesOrder[v1] = v2
            allNodes[idNoden]["childs"] = kids
            lefttrain = Xtrain[idxTrain][:, f] <= t
            righttrain = np.logical_not(lefttrain)
            lefttest = Xtest[idxTest][:, f] <= t
            righttest = np.logical_not(lefttest)
            stack.extend([(kids[1], idxTrain[righttrain], idxTest[righttest]),
                          (kids[0], idxTrain[lefttrain], idxTest[lefttest])])
    out_file = StringIO()
    out_file.write('digraph Tree {\n')
    out_file.write('labelloc="t";\n')
    out_file.write(('label="' + algo + ', Train AP: {:5.2f}'.format(
                    apTrain) + '%, Test AP: {:5.2f}'.format(apTest) + '%";\n'))
    out_file.write(('node [shape=box, style="filled", margin=0.01, ' +
                    'width=0, height=0] ;\n'))
    out_file.write(('graph [nodesep="0", ranksep="0.2", margin=0, ' +
                    'width=0, height=0];\n'))
    for node in allNodes.keys():
        out_file.write((str(node) + ' [label="'))
        if "node" not in allNodes[node]:
            out_file.write(('score = ' +
                            str(allNodes[node]["score"]) + '\\n'))
        else:
            f, t = allNodes[node]["node"]
            out_file.write(('X' + str(f) + " <= " + str(round(t, 3)) +
                            '\\n'))
        nptrain = allNodes[node]["nptrain"]
        nntrain = allNodes[node]["nntrain"]
        nptest = allNodes[node]["nptest"]
        nntest = allNodes[node]["nntest"]
        out_file.write(('train (' + str(nntrain) + ', ' + str(nptrain) + ')"'))
        if ("changeTree" in allNodes[node] and
           allNodes[node]["changeTree"] is True):
            out_file.write(', fillcolor="cyan"] ;\n')
        else:
            out_file.write(', fillcolor="white"] ;\n')
        if len(allNodes[node]["childs"]) == 2:
            c1, c2 = allNodes[node]["childs"]
            c1, c2 = strIdToNum[c1], strIdToNum[c2]
            if c1 == c2:
                out_file.write('%d -> %d [color="black"];\n' % (node, c1))
            else:
                out_file.write('%d -> %d [color="#6EAF0F"];\n' % (node, c1))
                out_file.write('%d -> %d [color="#ED5853"];\n' % (node, c2))
    for v1 in nodesOrder.keys():
        v2 = nodesOrder[v1]
        i1, i2 = strIdToNum[v1], strIdToNum[v2]
        out_file.write("{rank=same;%d->%d [style=invis];}\n" % (i1, i2))
    out_file.write("}")
    graph = graphviz.Source(out_file.getvalue())
    graph.render("figures/" + algo)


export_graphviz_TreeRank_MetaAP(clf.fullTree, Xtrain, ytrain, Xtest, ytest,
                                apTrain, apTest, "TreeRank")


def buildFullTree_MetaAP(clf):
    clf.fullTree = {}
    stack = [0]
    while len(stack) > 0:
        idNode = stack.pop()
        tree, leftLeaves = clf.trees[idNode]
        kids = clf.kids[idNode]
        if kids[1] in clf.trees:
            stack.append(kids[1])
        if kids[0] in clf.trees:
            stack.append(kids[0])
        stackTree = [0]
        tree = tree.tree_
        while len(stackTree) > 0:
            idTreeNode = stackTree.pop()
            curNode = str(idNode) + ";" + str(idTreeNode)
            if (tree.children_left[idTreeNode] == -1 and
               tree.children_right[idTreeNode] == -1):
                if idTreeNode in leftLeaves:
                    if (kids[0] not in clf.trees):
                        clf.fullTree[curNode] = {"score": clf.score[kids[0]]}
                    else:
                        clf.fullTree[curNode] = {"goToNode": str(kids[0])+";0"}
                else:
                    if (kids[1] not in clf.trees):
                        clf.fullTree[curNode] = {"score": clf.score[kids[1]]}
                    else:
                        clf.fullTree[curNode] = {"goToNode": str(kids[1])+";0"}
            else:
                kidsNode = [tree.children_right[idTreeNode],
                            tree.children_left[idTreeNode]]
                stackTree.extend(kidsNode)
                node = (tree.feature[idTreeNode],
                        tree.threshold.astype(np.float32)[idTreeNode],
                        str(idNode)+";"+str(kidsNode[1]),
                        str(idNode)+";"+str(kidsNode[0]))
                clf.fullTree[curNode] = {"node": node}
    nodes = []
    for curNode in clf.fullTree.keys():
        if "node" in clf.fullTree[curNode]:
            nodes.append(curNode)
    for curNode in nodes:
        f, t, k1, k2 = clf.fullTree[curNode]["node"]
        v1, v2 = k1, k2
        if "goToNode" in clf.fullTree[k1]:
            v1 = clf.fullTree[k1]["goToNode"]
            del clf.fullTree[k1]
            clf.fullTree[v1]["changeTree"] = True
        if "goToNode" in clf.fullTree[k2]:
            v2 = clf.fullTree[k2]["goToNode"]
            del clf.fullTree[k2]
            clf.fullTree[v2]["changeTree"] = True
        clf.fullTree[curNode]["node"] = (f, t, v1, v2)


clf = MetaAP(max_depth=2, bestresponse=1)
clf.fit(Xtrain, ytrain)
buildFullTree_MetaAP(clf)
rankTrain = clf.predict(Xtrain)
rankTest = clf.predict(Xtest)
apTrain, apTest = (average_precision_score(ytrain, rankTrain)*100,
                   average_precision_score(ytest, rankTest)*100)
print("MetaAP Train AP {:5.2f}".format(apTrain),
      "Test AP {:5.2f}".format(apTest))
export_graphviz_TreeRank_MetaAP(clf.fullTree, Xtrain, ytrain, Xtest, ytest,
                                apTrain, apTest, "MetaAP")
