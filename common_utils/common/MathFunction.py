import numpy as np
import math
import copy
import pandas as pd
# =============================================================================
#  delete values by index from dictionary 
#  SI is a dictionary
#  iD is the index for deleting
# =============================================================================   
def DicDelete(iD, SI):
    SI = copy.deepcopy(SI)
    L = len(iD)
    for i, j in SI.items():
        if j.shape[0] > L:
            j = np.delete(j, iD, 0)
        if j.shape[1] > L:
            j = np.delete(j, iD, 1)
        SI[i] = j

    return SI


# =============================================================================
#  The inverse matrices in a dictionary
#  D is a dictionary
# =============================================================================
def DicInv(D):
    D = copy.deepcopy(D)
    for i in D.keys():
        D[i] = 1 / D[i]
        D[i][D[i] == float('inf')] = 0
    return D


# =============================================================================
#  The deinition of norm for matrix M
# =============================================================================    
def norm_d(M, k):
    if k == 1:
        norm_ = np.mean(np.sum(abs(M), axis=0))
    elif k == 2:
        norm_ = np.mean(abs(M).sum(axis=0).sum(axis=1))
    elif k == 3:
        norm_ = math.sqrt(np.max(np.sum(np.power(M, 2), axis=0)))
    return norm_


# =============================================================================
#  The correlation coefficient 
# =============================================================================
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
    nn = np.shape(X)[1]
    ii_l = np.arange(0, nn, 1)
    i = 0
    while i < nn - 1:
        print([i, len(ii_l)])
        for j in range(nn - 1, i, -1):
            r2 = r_2(X[:, i], X[:, j])
            if r2 > r2_r or np.isnan(np.sum(X[:, j])) or np.sum(X[:, j]) == 0:
                ii_l = np.delete(ii_l, j, 0)
                X = np.delete(X, j, 1)
        i = i + 1
        nn = np.shape(X)[1]
    return ii_l


# =============================================================================
# delete the data with high duplication percent (p)
# ii_l is the index selected by data_dele
# =============================================================================    
def IndexDelete(X, ii_l, p):
    X = copy.deepcopy(X)
    ii_l = copy.deepcopy(ii_l)
    X = X[:, ii_l]
    L_X_S = np.shape(X)[1]
    X = pd.DataFrame(X)
    X_S_ = []
    for i in range(L_X_S):
        X_S = X.iloc[:, i].value_counts(normalize=True)
        X_S_max = max(X_S)
        X_S_.append(X_S_max)
    ii_l = ii_l[np.where(np.mat(X_S_) < p)[1]]
    return ii_l
