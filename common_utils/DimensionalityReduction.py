import copy
import math

import numpy as np

from model_utils.NewPreDeal_Tools import Del_deletion_data


def r_2(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    r2 = np.sum(np.multiply(x - x_mean, y - y_mean)) ** 2 / np.sum(np.power(x - x_mean, 2)) / np.sum(
        np.power(y - y_mean, 2))
    return r2


e = math.exp(1)


# =============================================================================
# delete the data with high correlation coefficient (r2_r)
# =============================================================================
def DataDelete(X, r2_r):
    X = copy.deepcopy(X)
    try:
        X=np.array(X,dtype="float64")
        X=Del_deletion_data(dataValue=X,flag=0)
        X=np.matrix(X)
        nn = np.shape(X)[1]
        ii_l = np.arange(0, nn, 1)
        i = 0
        while i < nn - 1:
            for j in range(nn - 1, i, -1):
                r2 = r_2(X[:, i], X[:, j])
                if r2 > r2_r or np.isnan(np.sum(X[:, j])) or np.sum(X[:, j]) == 0:
                    ii_l = np.delete(ii_l, j, 0)
                    X = np.delete(X, j, 1)
            i = i + 1
            nn = np.shape(X)[1]
        return ii_l
    except Exception as e:
        print("异常：",e)


if __name__ == '__main__':
    X = np.matrix([[1, 3, 3], [2, 6, 6], [3, 9, 9], [4, 12, 5]])
    ii_l = DataDelete(X, 0.90)
    x_1 = X[:, ii_l]
