import csv
import gzip
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from TreeRank import TreeRank
from SGBAP import SGBAP
os.utime("_tree.pxd", None)  # force recompilation of mytree used in MetaAP
os.utime("_tree.pyx", None)
from MetaAP import MetaAP, MetaAPForest
from ADT import ADT
from ADTRF import RandomForestClassifier as ADTRF


def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d


def oneHotEncodeColumns(data, columnsCategories):
    dataCategories = data[:, columnsCategories]
    dataEncoded = OneHotEncoder(sparse=False).fit_transform(dataCategories)
    columnsNumerical = []
    for i in range(data.shape[1]):
        if i not in columnsCategories:
            columnsNumerical.append(i)
    dataNumerical = data[:, columnsNumerical]
    return np.hstack((dataNumerical, dataEncoded)).astype(float)


def data_recovery(dataset):
    if dataset in ['abalone8', 'abalone17', 'abalone20']:
        data = pd.read_csv("datasets/abalone.data", header=None)
        data = pd.get_dummies(data, dtype=float)
        if dataset in ['abalone8']:
            y = np.array([1 if elt == 8 else 0 for elt in data[8]])
        elif dataset in ['abalone17']:
            y = np.array([1 if elt == 17 else 0 for elt in data[8]])
        elif dataset in ['abalone20']:
            y = np.array([1 if elt == 20 else 0 for elt in data[8]])
        X = np.array(data.drop([8], axis=1))
    elif dataset in ['autompg']:
        data = pd.read_csv("datasets/auto-mpg.data", header=None, sep=r'\s+')
        data = data.replace('?', np.nan)
        data = data.dropna()
        data = data.drop([8], axis=1)
        data = data.astype(float)
        y = np.array([1 if elt in [2, 3] else 0 for elt in data[7]])
        X = np.array(data.drop([7], axis=1))
    elif dataset in ['australian']:
        data, n, d = loadCsv('datasets/australian.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y != 1] = 0
    elif dataset in ['balance']:
        data = pd.read_csv("datasets/balance-scale.data", header=None)
        y = np.array([1 if elt in ['L'] else 0 for elt in data[0]])
        X = np.array(data.drop([0], axis=1))
    elif dataset in ['bankmarketing']:
        data, n, d = loadCsv('datasets/bankmarketing.csv')
        X = data[:, np.arange(0, d-1)]
        X = oneHotEncodeColumns(X, [1, 2, 3, 4, 6, 7, 8, 10, 15])
        y = data[:, d-1]
        y[y == "no"] = "0"
        y[y == "yes"] = "1"
        y = y.astype(int)
    elif dataset in ['bupa']:
        data, n, d = loadCsv('datasets/bupa.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y != 1] = 0
    elif dataset in ['german']:
        data = pd.read_csv("datasets/german.data-numeric", header=None,
                           sep=r'\s+')
        y = np.array([1 if elt == 2 else 0 for elt in data[24]])
        X = np.array(data.drop([24], axis=1))
    elif dataset in ['glass']:
        data = pd.read_csv("datasets/glass.data", header=None, index_col=0)
        y = np.array([1 if elt == 1 else 0 for elt in data[10]])
        X = np.array(data.drop([10], axis=1))
    elif dataset in ['hayes']:
        data = pd.read_csv("datasets/hayes-roth.data", header=None)
        y = np.array([1 if elt in [3] else 0 for elt in data[5]])
        X = np.array(data.drop([0, 5], axis=1))
    elif dataset in ['heart']:
        data, n, d = loadCsv('datasets/heart.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y = y.astype(int)
        y[y != 2] = 0
        y[y == 2] = 1
    elif dataset in ['iono']:
        data = pd.read_csv("datasets/ionosphere.data", header=None)
        y = np.array([1 if elt in ['b'] else 0 for elt in data[34]])
        X = np.array(data.drop([34], axis=1))
    elif dataset in ['libras']:
        data = pd.read_csv("datasets/movement_libras.data", header=None)
        y = np.array([1 if elt in [1] else 0 for elt in data[90]])
        X = np.array(data.drop([90], axis=1))
    elif dataset == "newthyroid":
        data, n, d = loadCsv('datasets/newthyroid.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y < 2] = 0
        y[y >= 2] = 1
    elif dataset in ['pageblocks']:
        data = pd.read_csv("datasets/page-blocks.data", header=None,
                           sep=r'\s+')
        y = np.array([1 if elt in [2, 3, 4, 5] else 0 for elt in data[10]])
        X = np.array(data.drop([10], axis=1))
    elif dataset in ['pima']:
        data, n, d = loadCsv('datasets/pima-indians-diabetes.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != '1'] = '0'
        y = y.astype(int)
    elif dataset in ['satimage']:
        data, n, d = loadCsv('datasets/satimage.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y = y.astype(int)
        y[y != 4] = 0
        y[y == 4] = 1
    elif dataset in ['segmentation']:
        data, n, d = loadCsv('datasets/segmentation.data')
        X = data[:, np.arange(1, d)].astype(float)
        y = data[:, 0]
        y[y == "WINDOW"] = '1'
        y[y != '1'] = '0'
        y = y.astype(int)
    elif dataset == "sonar":
        data, n, d = loadCsv('datasets/sonar.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != 'R'] = '0'
        y[y == 'R'] = '1'
        y = y.astype(int)
    elif dataset == "spambase":
        data, n, d = loadCsv('datasets/spambase.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y != 1] = 0
    elif dataset == "splice":
        data, n, d = loadCsv('datasets/splice.data')
        X = data[:, np.arange(1, d)].astype(float)
        y = data[:, 0].astype(int)
        y[y == 1] = 2
        y[y == -1] = 1
        y[y == 2] = 0
    elif dataset in ['vehicle']:
        data, n, d = loadCsv('datasets/vehicle.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != "van"] = '0'
        y[y == "van"] = '1'
        y = y.astype(int)
    elif dataset in ['wdbc']:
        data, n, d = loadCsv('datasets/wdbc.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != 'M'] = '0'
        y[y == 'M'] = '1'
        y = y.astype(int)
    elif dataset in ['wine']:
        data = pd.read_csv("datasets/wine.data", header=None)
        y = np.array([1 if elt == 1 else 0 for elt in data[0]])
        X = np.array(data.drop([0], axis=1))
    elif dataset in ['wine4']:
        data = pd.read_csv("datasets/winequality-red.csv", sep=';')
        y = np.array([1 if elt in [4] else 0 for elt in data.quality])
        X = np.array(data.drop(["quality"], axis=1))
    elif dataset in ['yeast3', 'yeast6']:
        data = pd.read_csv("datasets/yeast.data", header=None, sep=r'\s+')
        data = data.drop([0], axis=1)
        if dataset == 'yeast3':
            y = np.array([1 if elt == 'ME3' else 0 for elt in data[9]])
        elif dataset == 'yeast6':
            y = np.array([1 if elt == 'EXC' else 0 for elt in data[9]])
        X = np.array(data.drop([9], axis=1))
    return X, y


def listP(dic):  # Create grid of parameters given parameters ranges
    params = list(dic.keys())
    listParam = [{params[0]: value} for value in dic[params[0]]]
    for i in range(1, len(params)):
        newListParam = []
        currentParamName = params[i]
        currentParamRange = dic[currentParamName]
        for previousParam in listParam:
            for value in currentParamRange:
                newParam = previousParam.copy()
                newParam[currentParamName] = value
                newListParam.append(newParam)
        listParam = newListParam.copy()
    return listParam


def applyAlgo(algo, p, Xtrain, ytrain, Xtest, ytest, cv):
    if algo == "Gini" or algo == "Entropy":
        clf = ADT(max_depth=p["depth"], criterion=p["criterion"],
                  random_state=1)
        clf.fit(Xtrain, ytrain)
        rankTrain = clf.predict(Xtrain)
        rankTest = clf.predict(Xtest)
    if algo == "TreeRank":
        clf = TreeRank(max_depth=p["depth"])
        clf.fit(Xtrain, ytrain)
        rankTrain = clf.predict(Xtrain)
        rankTest = clf.predict(Xtest)
    if algo == "MetaAP":
        clf = MetaAP(max_depth=p["depth"], bestresponse=1)
        clf.fit(Xtrain, ytrain)
        rankTrain = clf.predict(Xtrain)
        rankTest = clf.predict(Xtest)
    if algo == "ForestEntropy":
        clf = ADTRF(max_depth=p["depth"], criterion=p["criterion"],
                    random_state=np.random.RandomState(1), n_estimators=100)
        clf.fit(Xtrain, ytrain)
        rankTrain = clf.predict(Xtrain)
        rankTest = clf.predict(Xtest)
    if algo == "MetaAPForest":
        clf = MetaAPForest(bestresponse=1, max_depth=p["depth"],
                           random_state=np.random.RandomState(1),
                           n_estimators=100)
        clf.fit(Xtrain, ytrain)
        rankTrain = clf.predict(Xtrain)
        rankTest = clf.predict(Xtest)
    if algo == "XGBRanker":
        clf = xgb.XGBRanker(max_depth=p["depth"], n_estimators=100,
                            learning_rate=p["learningRate"],
                            random_state=1, use_label_encoder=False)
        clf.fit(Xtrain, ytrain, group=[Xtrain.shape[0]])
        rankTrain = clf.predict(Xtrain)
        rankTest = clf.predict(Xtest)
    if algo == "SGBAP":
        clf = SGBAP(max_depth=p["depth"], n_estimators=100,
                    learning_rate=p["learningRate"])
        clf.fit(Xtrain, ytrain)
        rankTrain = clf.predict_proba(Xtrain)
        rankTest = clf.predict_proba(Xtest)
    apTrain, apTest = (average_precision_score(ytrain, rankTrain)*100,
                       average_precision_score(ytest, rankTest)*100)
    if cv is True:
        return apTrain, apTest
    else:
        ytrs = ytrain[np.argsort(rankTrain)[::-1]]
        ytes = ytest[np.argsort(rankTest)[::-1]]
        precisions = {}
        for pct in pcts:
            kTrain = max(1, int(pct * len(ytrain[ytrain == 1]) / 100))
            kTest = max(1, int(pct * len(ytest[ytest == 1]) / 100))
            ptrain = precision_score(ytrs[:kTrain], np.ones(kTrain))*100
            ptest = precision_score(ytes[:kTest], np.ones(kTest))*100
            precisions[pct] = (ptrain, ptest)
        return precisions, apTrain, apTest


nbFoldValid = 5
pcts = [1] + list(range(5, 101, 5))
seed = 1
if len(sys.argv) == 2:
    seed = int(sys.argv[1])
listParams = {"Gini": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40,
                                       50, 60, 70, 80, 90, 100],
                             "criterion": ["gini"]}),
              "Entropy": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30,
                                          40, 50, 60, 70, 80, 90, 100],
                                "criterion": ["entropy"]}),
              "TreeRank": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10]}),
              "MetaAP": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10]}),
              "ForestEntropy": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                      "criterion": ["entropy"]}),
              "MetaAPForest": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10]}),
              "XGBRanker": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                  "learningRate": [0.01, 0.05, 0.1, 0.2, 0.5,
                                                   0.75, 1]}),
              "SGBAP": listP({"depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                              "learningRate": [0.01, 0.05, 0.1, 0.2, 0.5,
                                               0.75, 1]})}
results = {}
for dataset in ['hayes', 'newthyroid', 'glass', 'bupa', 'wine', 'balance',
                'autompg', 'heart', 'pima', 'australian', 'yeast3',
                'yeast6', 'iono', 'sonar', 'vehicle', 'wdbc', 'wine4',
                'german', 'libras', 'abalone17', 'abalone20', 'abalone8',
                'segmentation', 'pageblocks', 'splice', 'satimage',
                'spambase', 'bankmarketing']:
    X, y = data_recovery(dataset)
    pctPos = 100*len(y[y == 1])/len(y)
    dataset = "{:05.2f}%".format(pctPos) + " " + dataset
    print(dataset, X.shape)
    np.random.seed(seed)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, shuffle=True,
                                                    stratify=y, test_size=0.3)
    skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
    foldsTrainValid = list(skf.split(Xtrain, ytrain))
    results[dataset] = {}
    for algo in listParams.keys():
        start = time.time()
        if len(listParams[algo]) > 1:  # Cross validation
            validParam = []
            for param in listParams[algo]:
                valid = []
                for iFoldVal in range(nbFoldValid):
                    fTrain, fValid = foldsTrainValid[iFoldVal]
                    valid.append(applyAlgo(algo, param,
                                           Xtrain[fTrain], ytrain[fTrain],
                                           Xtrain[fValid], ytrain[fValid],
                                           True)[1])
                validParam.append(np.mean(valid))
            param = listParams[algo][np.argmax(validParam)]
        else:  # No cross-validation
            param = listParams[algo][0]
        precisions, apTrain, apTest = applyAlgo(algo, param, Xtrain, ytrain,
                                                Xtest, ytest, False)
        results[dataset][algo] = (precisions, apTrain, apTest)
        print(dataset, algo, "Train AP {:5.2f}".format(apTrain),
              "Test AP {:5.2f}".format(apTest), param,
              "in {:6.2f}s".format(time.time()-start))
    if not os.path.exists("results"):
        try:
            os.makedirs("results")
        except:
            pass
    f = gzip.open("./results/res" + str(seed) + ".pklz", "wb")
    pickle.dump(results, f)
    f.close()
