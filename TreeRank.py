import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import gc
base = importr('base')
robjects.r('''
library(rpart)
source("./TreeRank.R")
''')


def dataframePythonToR(frame):
    with localconverter(robjects.default_converter + pandas2ri.converter) as c:
        frame_R = c.py2rpy(frame)
    return frame_R


class TreeRank():
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __del__(self):
        del self.tree
        gc.collect()
        robjects.r.gc()

    def fit(self, X, y):
        X_R = dataframePythonToR(pd.DataFrame(X))
        y_R = dataframePythonToR(pd.DataFrame(data=y, columns=["label"]))
        TreeRank = robjects.globalenv['TreeRank']
        self.tree = TreeRank(formula="label~ .", data=base.cbind(X_R, y_R),
                             bestresponse=1, minsplit=2,
                             maxdepth=self.max_depth, criterion="TR")

    def predict(self, X):
        X_R = dataframePythonToR(pd.DataFrame(X))
        predict = robjects.globalenv['predictTR_TreeRank']
        return np.array(predict(self.tree, X_R))
