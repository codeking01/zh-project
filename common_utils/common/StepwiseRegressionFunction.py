import numpy as np
from itertools import combinations
import copy
import warnings
from scipy import optimize as op
from numpy import mat as mat

cf = op.curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# rf=RandomForestRegressor()
# 分类
rf = RandomForestClassifier()

reg = LinearRegression()
warnings.filterwarnings("ignore")


# import NoneLineFunction as nlf

# =============================================================================
# statistic: statistic parameter
# r2: correlation coefficient
# aae: average absolute error
# aad: average absolute deviation
# rmse: root mean square error
# x: calculated value
# y: observed value
# I: 1 or -1
# =============================================================================
class statistic:
    # def r2(x, y, I):
    #     x_mean = np.mean(x)
    #     y_mean = np.mean(y)
    #     r2 = np.sum(np.multiply(x - x_mean, y - y_mean)) ** 2 / np.sum(np.power(x - x_mean, 2)) / np.sum(
    #         np.power(y - y_mean, 2))
    #     return r2
    def r2(x, y,I):
        from sklearn.metrics import accuracy_score
        return accuracy_score(x,y)

    def aae(x, y, I):
        aae = I * np.mean(np.abs(x - y))
        return aae

    def rmse(x, y, I):
        rmse = I * np.sqrt(np.mean(np.multiply(x - y, x - y)))
        return rmse

    def aad(x, y, I):
        aad = I * np.mean(np.abs((x - y) / y))
        return aad

    def total(x, y, I):
        return [statistic.r2(x, y, 1), statistic.aae(x, y, -1), statistic.rmse(x, y, -1)]


# =============================================================================
# evaluation: evaluation method
# evl_i: evaluate by one statistic parameter
# evl_total: evaluate by multiple statistic parameters
# =============================================================================
class evaluation:
    def evl_p(res_):
        res_s = copy.deepcopy(res_)
        res_s.sort(0)
        L = np.shape(res_)[0]
        res_s_ = res_s[round(L * 0.05), :]
        return res_s_

    def evl_i(res_):
        res_ = mat(res_)
        res_abs = abs(res_)
        res_s_ = evaluation.evl_p(res_)
        SS_w = np.where((res_ > res_s_[0, 0]) & (np.sum(res_abs, 1) != np.inf) & (~np.isnan(np.sum(res_abs, 1))))[0]
        res_ = res_[SS_w, :]
        i_max = max(res_)
        i_ = np.where(res_ == i_max)[0][0]
        i = SS_w[i_]
        return i

    def evl_total(res_):
        res_ = mat(res_)
        res_abs = abs(res_)
        res_s_ = evaluation.evl_p(res_)
        SS_w = np.where((res_[:, 0] > res_s_[0, 0]) & (res_[:, 1] > res_s_[0, 1]) & (res_[:, 2] > res_s_[0, 2]) & (
                    np.sum(res_abs, 1) != np.inf) & (~np.isnan(np.sum(res_abs, 1))))[0]
        res_ = res_[SS_w, :]
        _mean = np.mean(res_, 0)
        _std = np.std(res_, 0)
        res_ = (res_ - _mean) / _std
        _min = np.min(res_, 0)
        _max = np.max(res_, 0)
        res_max = np.sqrt(np.sum(np.multiply(res_ - _max, res_ - _max), 1))
        res_min = np.sqrt(np.sum(np.multiply(res_ - _min, res_ - _min), 1))
        all_max_ = res_max / (res_min + res_max)
        all_min = min(all_max_)
        i_ = np.where(all_max_ == all_min)[0][0]
        i = SS_w[i_]
        return i


class regress:
    def line(x, y):
        beta = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
        beta = beta.T.tolist()[0]
        return beta

    def line_sklearn(x, y):
        reg.fit(x[:, 1:], y)
        beta = np.c_[reg.intercept_, reg.coef_]
        beta = beta.tolist()[0]
        return beta

    def non_line(x, y):
        y_ = np.asarray(y.T)
        y_ = y_[0]
        x_ = np.asarray(x.T)
        beta0 = np.asarray(np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), nlf.fy(y)))
        beta = cf(nlf.f, x_, y_, beta0, maxfev=10000)[0]
        beta = beta.T.tolist()
        return beta

    def non_line_sklearn(x, y):
        reg.fit(x[:, 1:], y)
        beta0 = np.c_[reg.intercept_, reg.coef_]
        y_ = np.asarray(y.T)
        y_ = y_[0]
        x_ = np.asarray(x.T)
        beta = cf(nlf.f, x_, y_, beta0, maxfev=10000)[0]
        beta = beta.T.tolist()
        return beta

    def random_forest(x, y):
        rf.fit(x[:, 1:], y)
        return rf


class calculation:
    def line(x, beta):
        beta = mat(beta).T
        y = x * beta
        return y

    def line_sklearn(x, beta):
        beta = mat(beta).T
        y = beta[0, :] + x[:, 1:] * beta[1:, :]
        return y

    def non_line(x, beta):
        x = np.asarray(x.T)
        beta = np.asarray(beta)
        y = mat(nlf.f(x, beta)).T
        return y

    def non_line_sklearn(x, beta):
        x = np.asarray(x[:, 1:].T)
        beta = np.asarray(beta)
        y = mat(nlf.f(x, beta)).T
        return y

    def random_forest(x, rf):
        y = mat(rf.predict(x[:, 1:])).T
        return y


class QSPR:
    # =============================================================================
    # ForwardExternal: forward stepwise regression evaluated by external validation
    # =============================================================================
    def ForwardExternal(regr, k_ad, kk_ad_all, evl, Y_train, Y_test, ii, XT_train, XT_test, XT0_train, XT0_test):
        sta = getattr(statistic, evl)
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        n = np.shape(XT_train)[1]
        c = list(combinations(np.arange(0, n, 1), k_ad))
        for kk in range(0, kk_ad_all):
            XT_train_ii = np.c_[XT0_train, XT_train[:, ii]]
            XT_test_ii = np.c_[XT0_test, XT_test[:, ii]]
            res_ = []
            jj_ = []
            bb_ = []
            for jj in c:
                x_train = np.c_[XT_train_ii, XT_train[:, jj]]
                x_test = np.c_[XT_test_ii, XT_test[:, jj]]
                try:
                    beta = reg(x_train, Y_train)
                    Y_cal_train = cal(x_train, beta)
                    Y_cal_test = cal(x_test, beta)
                    res_train = sta(x=Y_cal_train, y=Y_train, I=-1)
                    res_test = sta(Y_cal_test, Y_test, -1)
                    res_.append(np.min(np.r_[mat(res_train), mat(res_test)], 0).tolist()[0])
                    jj_.append(jj)
                    bb_.append(beta)
                except:
                    pass
            if evl == 'total':
                i_ = evaluation.evl_total(res_)
            else:
                i_ = evaluation.evl_i(res_)
            ii = ii + list(jj_[i_])
            bb = bb_[i_]
        return ii, bb

    # =============================================================================
    # BackwardExternal: backward stepwise regression evaluated by external validation
    # =============================================================================
    def BackwardExternal(regr, s, n_de_all, evl, Y_train, Y_test, ii, XT_train, XT_test, XT0_train, XT0_test):
        sta = getattr(statistic, evl)
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        if s == 'total':
            n_end = 1
            c_end = [n_de_all]
        else:
            n_end = int(np.ceil(n_de_all / s))
            c_end = (s * np.ones((1, n_end - 1))[0]).astype(int).tolist() + [int(n_de_all - s * (n_end - 1))]
        for kk in c_end:
            n = len(ii)
            c = list(combinations(np.arange(0, n, 1), kk))
            XL_train = XT_train[:, ii]
            XL_test = XT_test[:, ii]
            i_ = np.arange(0, n, 1)
            res_ = []
            ii_ = []
            bb_ = []
            for jj in c:
                i = np.arange(0, n, 1)
                i = np.delete(copy.deepcopy(i_), jj, 0)
                x_train = np.c_[XT0_train, XL_train[:, i]]
                x_test = np.c_[XT0_test, XL_test[:, i]]
                try:
                    beta = reg(x_train, Y_train)
                    Y_cal_train = cal(x_train, beta)
                    Y_cal_test = cal(x_test, beta)
                    res_train = sta(Y_cal_train, Y_train, -1)
                    res_test = sta(Y_cal_test, Y_test, -1)
                    res_.append(np.min(np.r_[mat(res_train), mat(res_test)], 0).tolist()[0])
                    ii_.append(i)
                    bb_.append(beta)
                except:
                    pass
            if evl == 'total':
                i_ = evaluation.evl_total(res_)
            else:
                i_ = evaluation.evl_i(res_)
            ii = np.array(ii)[ii_[i_]].tolist()
            bb = bb_[i_]
        return ii, bb

    # =============================================================================
    # status: the status of the model
    # =============================================================================
    def status(regr, Y_train, Y_test, XT_train, XT_test, ii, XT0_train, XT0_test, beta):
        cal = getattr(calculation, regr)
        x_train = np.c_[XT0_train, XT_train[:, ii]]
        x_test = np.c_[XT0_test, XT_test[:, ii]]
        Y_cal_train = cal(x_train, beta)
        Y_cal_test = cal(x_test, beta)
        Y_cal_all = np.r_[Y_cal_train, Y_cal_test]
        Y_all = np.r_[Y_train, Y_test]
        # 拟合
        # r2_train = statistic.r2(Y_cal_train, Y_train, 1)
        # r2_test = statistic.r2(Y_cal_test, Y_test, 1)
        # r2 = statistic.r2(Y_all, Y_cal_all, 1)
        # aae_train = statistic.aae(Y_cal_train, Y_train, 1)
        # aae_test = statistic.aae(Y_cal_test, Y_test, 1)
        # aae = statistic.aae(Y_cal_all, Y_all, 1)
        # aad_train = statistic.aad(Y_cal_train, Y_train, 1)
        # aad_test = statistic.aad(Y_cal_test, Y_test, 1)
        # aad = statistic.aad(Y_cal_all, Y_all, 1)

        # 分类的
        from sklearn.metrics import accuracy_score
        r2_train = accuracy_score(Y_cal_train, Y_train)
        r2_test = accuracy_score(Y_cal_test, Y_test)
        r2 = accuracy_score(Y_all, Y_cal_all)
        aae_train = 0
        aae_test = 0
        aae = 0
        aad_train = 0
        aad_test = 0
        aad = 0
        return r2, aae, aad, r2_train, aae_train, aad_train, r2_test, aae_test, aad_test

    # =============================================================================
    # status: the status of the model
    # =============================================================================
    def status_regress(regr, Y_train, Y_test, XT_train, XT_test, ii, XT0_train, XT0_test):
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        x_train = np.c_[XT0_train, XT_train[:, ii]]
        x_test = np.c_[XT0_test, XT_test[:, ii]]
        beta = reg(x_train, Y_train)
        Y_cal_train = cal(x_train, beta)
        Y_cal_test = cal(x_test, beta)
        Y_cal_all = np.r_[Y_cal_train, Y_cal_test]
        Y_all = np.r_[Y_train, Y_test]
        r2_train = statistic.r2(Y_cal_train, Y_train, 1)
        r2_test = statistic.r2(Y_cal_test, Y_test, 1)
        r2 = statistic.r2(Y_all, Y_cal_all, 1)
        aae_train = statistic.aae(Y_cal_train, Y_train, 1)
        aae_test = statistic.aae(Y_cal_test, Y_test, 1)
        aae = statistic.aae(Y_cal_all, Y_all, 1)
        aad_train = statistic.aad(Y_cal_train, Y_train, 1)
        aad_test = statistic.aad(Y_cal_test, Y_test, 1)
        aad = statistic.aad(Y_cal_all, Y_all, 1)
        return beta, r2, aae, aad, r2_train, aae_train, aad_train, r2_test, aae_test, aad_test

    # =============================================================================
    # C_fold: leave-one-component-out cross-validation
    # =============================================================================
    def C_fold(regr, XLA, Y, ii, XT, XT0):
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        YL_A = np.zeros(np.shape(Y))
        for jj in range(0, max(XLA)[0, 0] + 1):
            ss = np.where(XLA == jj)[0]
            if len(ss) > 0:
                YL = Y
                XTL = XT[:, ii]
                XTLQ = copy.deepcopy(XTL[ss, :])
                XTL = np.delete(XTL, ss, 0)
                XTLQ0 = copy.deepcopy(XT0[ss, :])
                XTL0 = np.delete(XT0, ss, 0)
                YL = np.delete(YL, ss, 0)
                x_train = np.c_[XTL0, XTL]
                x_test = np.c_[XTLQ0, XTLQ]
                beta = reg(x_train, YL)
                Y_cal_test = cal(x_test, beta)
                YL_A[ss, :] = Y_cal_test
        r2_0 = statistic.r2(YL_A, Y, 1)
        aae = statistic.aae(YL_A, Y, 1)
        aad = statistic.aad(YL_A, Y, 1)
        rmse = statistic.rmse(YL_A, Y, 1)
        return r2_0, aae, aad, rmse, YL_A

    # =============================================================================
    # loocv: leave-one-data-out cross-validation
    # =============================================================================
    def loocv(regr, Y, ii, XT, XT0):
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        YL_cal = np.zeros(np.shape(Y))
        for jj in range(0, len(Y)):
            YL = Y
            XTL = XT[:, ii]
            XTLQ = copy.deepcopy(XTL[jj, :])
            XTL = np.delete(XTL, jj, 0)
            XTLQ0 = copy.deepcopy(XT0[jj, :])
            XTL0 = np.delete(XT0, jj, 0)
            YL = np.delete(YL, jj, 0)
            x_train = np.c_[XTL0, XTL]
            x_test = np.c_[XTLQ0, XTLQ]
            beta = reg(x_train, YL)
            Y_cal_test = cal(x_test, beta)
            YL_cal[jj, :] = Y_cal_test
        r2_0 = statistic.r2(YL_cal, Y, 1)
        aae = statistic.aae(YL_cal, Y, 1)
        aad = statistic.aad(YL_cal, Y, 1)
        rmse = statistic.rmse(YL_cal, Y, 1)
        return r2_0, aae, aad, rmse


class GroupContribution:
    # =============================================================================
    # GroupCombination
    # g_s is the index of selected group
    # g is group
    # =============================================================================
    def GroupCombination(g_s, group):
        L = len(group)
        g_key = list(group.keys())
        gc = np.zeros(np.shape(group[g_key[0]]))
        for i in range(0, L):
            ww = np.where(g_s == i)[1]
            gc[:, ww] = group[g_key[i]][:, ww]
        return gc

    # =============================================================================
    # GroupSelectionExternal: select group by external validation
    # g_a is the index of selected group
    # g is group
    # =============================================================================
    def GroupSelectionExternal(regr, evl, g_train, g_test, g_s, Y_train, Y_test, ii_r, ii_c, XT_train, XT_test,
                               XT0_train, XT0_test, R, Cg):
        sta = getattr(statistic, evl)
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        g_key = list(g_train.keys())
        gs_0_train = np.zeros(np.shape(g_train[g_key[0]]))
        gs_0_test = np.zeros(np.shape(g_test[g_key[0]]))
        ii = np.array(ii_c) + np.array(ii_r) * R
        xt_train = XT_train[:, ii]
        xt_test = XT_test[:, ii]
        L_g_s = np.shape(g_s)[1]
        ij = 1
        r2_0 = -1E10
        r2 = -1E9
        while ij != 0 and r2 > r2_0:
            r2_0 = r2
            g_s_0 = copy.deepcopy(g_s)
            res_ = []
            ii_t = []
            bb_ = []
            for mm in range(0, L_g_s):
                for nn in range(0, Cg):
                    g_s_ = copy.deepcopy(g_s)
                    g_s_[0, mm] = nn
                    gs_t_train = copy.deepcopy(gs_0_train)
                    gs_t_test = copy.deepcopy(gs_0_test)
                    for kk in range(0, Cg):
                        ww = np.where(g_s_ == kk)[1]
                        gs_t_train[:, ww] = g_train[g_key[kk]][:, ww]
                        gs_t_test[:, ww] = g_test[g_key[kk]][:, ww]
                    x_train = np.c_[gs_t_train, xt_train]
                    x_test = np.c_[gs_t_test, xt_test]
                    try:
                        X_train = np.c_[XT0_train, x_train]
                        X_test = np.c_[XT0_test, x_test]
                        beta = reg(X_train, Y_train)
                        Y_cal_train = cal(X_train, beta)
                        Y_cal_test = cal(X_test, beta)
                        res_train = sta(x=Y_cal_train, y=Y_train, I=-1)
                        res_test = sta(Y_cal_test, Y_test, -1)
                        res_.append(np.min(np.r_[mat(res_train), mat(res_test)], 0).tolist()[0])
                        ii_t.append([mm, nn])
                        bb_.append(beta)
                    except:
                        pass
            if evl == 'total':
                i_ = evaluation.evl_total(res_)
            else:
                i_ = evaluation.evl_i(res_)
            g_s[0, ii_t[i_][0]] = ii_t[i_][1]
            ij = np.sum(abs(g_s_0 - g_s), 1)[0]
            r2 = res_[i_][0]
            bb = bb_[i_]
        return g_s, bb

    # =============================================================================
    # GroupSelectionExternal: select group by external validation
    # g_a is the index of selected group
    # g is group
    # =============================================================================
    def GroupSelectionExternal0(regr, evl, g_train, g_test, g_s, Y_train, Y_test, XT0_train, XT0_test, Cg):
        sta = getattr(statistic, evl)
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        g_key = list(g_train.keys())
        gs_0_train = np.zeros(np.shape(g_train[g_key[0]]))
        gs_0_test = np.zeros(np.shape(g_test[g_key[0]]))
        L_g_s = np.shape(g_s)[1]
        ij = 1
        r2_0 = -1E10
        r2 = -1E9
        while ij != 0 and r2 > r2_0:
            r2_0 = r2
            g_s_0 = copy.deepcopy(g_s)
            res_ = []
            ii_t = []
            bb_ = []
            for mm in range(0, L_g_s):
                for nn in range(0, Cg):
                    g_s_ = copy.deepcopy(g_s)
                    g_s_[0, mm] = nn
                    gs_t_train = copy.deepcopy(gs_0_train)
                    gs_t_test = copy.deepcopy(gs_0_test)
                    for kk in range(0, Cg):
                        ww = np.where(g_s_ == kk)[1]
                        gs_t_train[:, ww] = g_train[g_key[kk]][:, ww]
                        gs_t_test[:, ww] = g_test[g_key[kk]][:, ww]
                    try:
                        X_train = np.c_[XT0_train, gs_t_train]
                        X_test = np.c_[XT0_test, gs_t_test]

                        beta = reg(X_train, Y_train)
                        Y_cal_train = cal(X_train, beta)
                        Y_cal_test = cal(X_test, beta)

                        res_train = sta(x=Y_cal_train, y=Y_train, I=-1)
                        res_test = sta(Y_cal_test, Y_test, -1)
                        res_.append(np.min(np.r_[mat(res_train), mat(res_test)], 0).tolist()[0])
                        ii_t.append([mm, nn])
                        bb_.append(beta)
                    except:
                        pass
            if evl == 'total':
                i_ = evaluation.evl_total(res_)
            else:
                i_ = evaluation.evl_i(res_)
            g_s[0, ii_t[i_][0]] = ii_t[i_][1]
            ij = np.sum(abs(g_s_0 - g_s), 1)[0]
            r2 = res_[i_][0]
            bb = bb_[i_]
        return g_s, bb

    # =============================================================================
    # ForwardExternal: forward stepwise regression evaluated by external validation
    # =============================================================================
    def ForwardExternal(regr, kk_ad_all, evl, Y_train, Y_test, ii_r, ii_c, XT_train, XT_test, XT0_train, XT0_test,
                        gs_train, gs_test, R, C):
        sta = getattr(statistic, evl)
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        for kk in range(0, kk_ad_all):
            # 增加定位因子
            ii = np.array(ii_c) + np.array(ii_r) * R
            ii = ii.tolist()
            jj_t = np.arange(0, R, 1)
            jj_t = np.delete(jj_t, ii_c, 0)
            XT_train_ii = np.c_[gs_train, XT_train[:, ii]]
            XT_test_ii = np.c_[gs_test, XT_test[:, ii]]
            res_ = []
            ii_t = []
            bb_ = []
            for jj in jj_t:
                for j in range(0, C):
                    i = [jj + j * R]
                    x_train = np.c_[XT_train_ii, XT_train[:, i]]
                    x_test = np.c_[XT_test_ii, XT_test[:, i]]
                    try:
                        X_train = np.c_[XT0_train, x_train]
                        X_test = np.c_[XT0_test, x_test]
                        beta = reg(X_train, Y_train)
                        Y_cal_train = cal(X_train, beta)
                        Y_cal_test = cal(X_test, beta)
                        res_train = sta(x=Y_cal_train, y=Y_train, I=-1)
                        res_test = sta(Y_cal_test, Y_test, -1)
                        res_.append(np.min(np.r_[mat(res_train), mat(res_test)], 0).tolist()[0])
                        ii_t.append([jj, j])
                        bb_.append(beta)
                    except:
                        pass
            if evl == 'total':
                i_ = evaluation.evl_total(res_)
            else:
                i_ = evaluation.evl_i(res_)
            ii_c.append(ii_t[i_][0])
            ii_r.append(ii_t[i_][1])
            bb = bb_[i_]
            # 增加定位因子后，对定位因子进行重新选择
            ii_r, bb = GroupContribution.EvaluatepositionExternal(regr, evl, Y_train, Y_test, ii_r, ii_c, XT_train,
                                                                  XT_test, XT0_train, XT0_test, gs_train, gs_test, R, C)
        return ii_r, ii_c, bb

    # =============================================================================
    # ForwardExternal: forward stepwise regression evaluated by external validation
    # =============================================================================
    def EvaluatepositionExternal(regr, evl, Y_train, Y_test, ii_r, ii_c, XT_train, XT_test, XT0_train, XT0_test,
                                 gs_train, gs_test, R, C):
        sta = getattr(statistic, evl)
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        ii_r0 = []
        kk = 0
        while ii_r0 != ii_r and kk < 20:
            kk += 1
            ii_r0 = copy.deepcopy(ii_r)
            res_ = []
            ii_t = []
            bb_ = []
            for jj in range(0, len(ii_r)):
                for j in range(0, C):
                    ii_r_t = copy.deepcopy(ii_r)
                    ii_r_t[jj] = j
                    i = np.array(ii_c) + np.array(ii_r_t) * R
                    x_train = np.c_[gs_train, XT_train[:, i]]
                    x_test = np.c_[gs_test, XT_test[:, i]]
                    try:
                        X_train = np.c_[XT0_train, x_train]
                        X_test = np.c_[XT0_test, x_test]
                        beta = reg(X_train, Y_train)
                        Y_cal_train = cal(X_train, beta)
                        Y_cal_test = cal(X_test, beta)
                        res_train = sta(x=Y_cal_train, y=Y_train, I=-1)
                        res_test = sta(Y_cal_test, Y_test, -1)
                        res_.append(np.min(np.r_[mat(res_train), mat(res_test)], 0).tolist()[0])
                        ii_t.append([jj, j])
                        bb_.append(beta)
                    except:
                        pass
            if evl == 'total':
                i_ = evaluation.evl_total(res_)
            else:
                i_ = evaluation.evl_i(res_)
            ii_r[ii_t[i_][0]] = ii_t[i_][1]
            bb = bb_[i_]
        return ii_r, bb

    # =============================================================================
    # BackwardExternal: backward stepwise regression evaluated by external validation
    # =============================================================================
    def BackwardExternal(regr, s, kk_de_all, evl, Y_train, Y_test, ii_r, ii_c, XT_train, XT_test, XT0_train, XT0_test,
                         gs_train, gs_test, R, C):
        sta = getattr(statistic, evl)
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        kk_end = int(np.ceil(kk_de_all / s))
        c_end = (s * np.ones((1, kk_end - 1))[0]).astype(int).tolist() + [int(kk_de_all - s * (kk_end - 1))]
        for kk in range(0, kk_end):
            L_c = len(ii_c)
            c = list(combinations(np.arange(0, L_c, 1), c_end[kk]))
            res_ = []
            ii_ = []
            bb_ = []
            for jj in c:
                ii_r_t = copy.deepcopy(ii_r)
                ii_r_t = np.delete(ii_r_t, jj)
                ii_c_t = copy.deepcopy(ii_c)
                ii_c_t = np.delete(ii_c_t, jj)
                i = np.array(ii_c_t) + np.array(ii_r_t) * R
                x_train = np.c_[gs_train, XT_train[:, i]]
                x_test = np.c_[gs_test, XT_test[:, i]]
                try:
                    X_train = np.c_[XT0_train, x_train]
                    X_test = np.c_[XT0_test, x_test]
                    beta = reg(X_train, Y_train)
                    Y_cal_train = cal(X_train, beta)
                    Y_cal_test = cal(X_test, beta)
                    res_train = sta(x=Y_cal_train, y=Y_train, I=-1)
                    res_test = sta(Y_cal_test, Y_test, -1)
                    res_.append(np.min(np.r_[mat(res_train), mat(res_test)], 0).tolist()[0])
                    ii_.append(jj)
                    bb_.append(beta)
                except:
                    pass
            if evl == 'total':
                i_ = evaluation.evl_total(res_)
            else:
                i_ = evaluation.evl_i(res_)
            ii_r = list(np.delete(ii_r, ii_[i_]))
            ii_c = list(np.delete(ii_c, ii_[i_]))
            bb = bb_[i_]
            # 减少定位因子后，对定位因子进行重新选择
            ii_r, bb = GroupContribution.EvaluatepositionExternal(regr, evl, Y_train, Y_test, ii_r, ii_c, XT_train,
                                                                  XT_test, XT0_train, XT0_test, gs_train, gs_test, R, C)
        return ii_r, ii_c, bb

    # =============================================================================
    # status: the status of the model
    # =============================================================================

    def status(regr, Y_train, Y_test, ii_r, ii_c, XT_train, XT_test, XT0_train, XT0_test, gs_train, gs_test, R, beta):
        cal = getattr(calculation, regr)
        ii = np.array(ii_c) + np.array(ii_r) * R
        x_train = np.c_[gs_train, XT_train[:, ii]]
        x_test = np.c_[gs_test, XT_test[:, ii]]
        X_train = np.c_[XT0_train, x_train]
        X_test = np.c_[XT0_test, x_test]
        Y_cal_train = cal(X_train, beta)
        Y_cal_test = cal(X_test, beta)

        Y_cal_all = np.r_[Y_cal_train, Y_cal_test]
        Y_all = np.r_[Y_train, Y_test]
        r2_train = statistic.r2(Y_cal_train, Y_train, 1)
        r2_test = statistic.r2(Y_cal_test, Y_test, 1)
        r2 = statistic.r2(Y_all, Y_cal_all, 1)
        aad_train = statistic.aad(Y_cal_train, Y_train, 1)
        aad_test = statistic.aad(Y_cal_test, Y_test, 1)
        aad = statistic.aad(Y_cal_all, Y_all, 1)
        aae_train = statistic.aae(Y_cal_train, Y_train, 1)
        aae_test = statistic.aae(Y_cal_test, Y_test, 1)
        aae = statistic.aae(Y_cal_all, Y_all, 1)
        return r2, aad, aae, r2_train, aad_train, aae_train, r2_test, aad_test, aae_test

    # =============================================================================
    # status: the status of the model
    # =============================================================================

    def status_regress(regr, Y_train, Y_test, ii_r, ii_c, XT_train, XT_test, XT0_train, XT0_test, gs_train, gs_test, R):
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        ii = np.array(ii_c) + np.array(ii_r) * R
        x_train = np.c_[gs_train, XT_train[:, ii]]
        x_test = np.c_[gs_test, XT_test[:, ii]]
        X_train = np.c_[XT0_train, x_train]
        X_test = np.c_[XT0_test, x_test]
        beta = reg(X_train, Y_train)
        Y_cal_train = cal(X_train, beta)
        Y_cal_test = cal(X_test, beta)

        Y_cal_all = np.r_[Y_cal_train, Y_cal_test]
        Y_all = np.r_[Y_train, Y_test]
        r2_train = statistic.r2(Y_cal_train, Y_train, 1)
        r2_test = statistic.r2(Y_cal_test, Y_test, 1)
        r2 = statistic.r2(Y_all, Y_cal_all, 1)
        aad_train = statistic.aad(Y_cal_train, Y_train, 1)
        aad_test = statistic.aad(Y_cal_test, Y_test, 1)
        aad = statistic.aad(Y_cal_all, Y_all, 1)
        aae_train = statistic.aae(Y_cal_train, Y_train, 1)
        aae_test = statistic.aae(Y_cal_test, Y_test, 1)
        aae = statistic.aae(Y_cal_all, Y_all, 1)
        return beta, r2, aad, aae, r2_train, aad_train, aae_train, r2_test, aad_test, aae_test

    # =============================================================================
    # status: the status of the model
    # =============================================================================

    def status0(regr, Y_train, Y_test, gs_train, gs_test, XT0_train, XT0_test, beta):
        cal = getattr(calculation, regr)
        X_train = np.c_[XT0_train, gs_train]
        X_test = np.c_[XT0_test, gs_test]
        Y_cal_train = cal(X_train, beta)
        Y_cal_test = cal(X_test, beta)
        Y_cal_all = np.r_[Y_cal_train, Y_cal_test]
        Y_all = np.r_[Y_train, Y_test]
        r2_train = statistic.r2(Y_cal_train, Y_train, 1)
        r2_test = statistic.r2(Y_cal_test, Y_test, 1)
        r2 = statistic.r2(Y_all, Y_cal_all, 1)
        aad_train = statistic.aad(Y_cal_train, Y_train, 1)
        aad_test = statistic.aad(Y_cal_test, Y_test, 1)
        aad = statistic.aad(Y_cal_all, Y_all, 1)
        aae_train = statistic.aae(Y_cal_train, Y_train, 1)
        aae_test = statistic.aae(Y_cal_test, Y_test, 1)
        aae = statistic.aae(Y_cal_all, Y_all, 1)
        return r2, aad, aae, r2_train, aad_train, aae_train, r2_test, aad_test, aae_test

    # =============================================================================
    # C_fold: leave-one-component-out cross-validation
    # =============================================================================
    def C_fold(regr, XLA, Y, ii_r, ii_c, XT, XT0, gs, R):
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        ii = np.array(ii_c) + np.array(ii_r) * R
        YL_cal = np.zeros(np.shape(Y))
        for jj in XLA:
            ss = np.where(XLA == jj[0, 0])[0]
            YL = Y
            XTL = np.c_[gs, XT[:, ii]]
            XTLQ = copy.deepcopy(XTL[ss, :])
            XTL = np.delete(XTL, ss, 0)
            XTLQ0 = copy.deepcopy(XT0[ss, :])
            XTL0 = np.delete(XT0, ss, 0)
            YL = np.delete(YL, ss, 0)
            x_train = np.c_[XTL0, XTL]
            x_test = np.c_[XTLQ0, XTLQ]
            beta = reg(x_train, YL)
            Y_cal_0 = cal(x_test, beta)
            YL_cal[ss, :] = Y_cal_0
        r2_0 = statistic.r2(YL_cal, Y, 1)
        aae = statistic.aae(YL_cal, Y, 1)
        aad = statistic.aad(YL_cal, Y, 1)
        rmse = statistic.rmse(YL_cal, Y, 1)
        return r2_0, aae, aad, rmse, YL_cal

    # =============================================================================
    # C_fold: leave-one-component-out cross-validation
    # =============================================================================
    def C_fold0(regr, XLA, Y, XT0, gs):
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        YL_cal = np.zeros(np.shape(Y))
        for jj in XLA:
            ss = np.where(XLA == jj[0, 0])[0]
            YL = Y
            XTL = gs
            XTLQ = copy.deepcopy(XTL[ss, :])
            XTL = np.delete(XTL, ss, 0)
            XTLQ0 = copy.deepcopy(XT0[ss, :])
            XTL0 = np.delete(XT0, ss, 0)
            YL = np.delete(YL, ss, 0)
            x_train = np.c_[XTL0, XTL]
            x_test = np.c_[XTLQ0, XTLQ]
            beta = reg(x_train, YL)
            Y_cal_0 = cal(x_test, beta)
            YL_cal[ss, :] = Y_cal_0
        r2_0 = statistic.r2(YL_cal, Y, 1)
        aae = statistic.aae(YL_cal, Y, 1)
        aad = statistic.aad(YL_cal, Y, 1)
        rmse = statistic.rmse(YL_cal, Y, 1)
        return r2_0, aae, aad, rmse, YL_cal

    # =============================================================================
    # C_fold: leave-one-component-out cross-validation
    # =============================================================================
    def C_fold1(regr, XLA, Y, ii_r, ii_c, XT, XT0, gs, R):
        reg = getattr(regress, regr)
        cal = getattr(calculation, regr)
        ii = np.array(ii_c) + np.array(ii_r) * R
        YL_cal = np.zeros(np.shape(Y))
        for jj in XLA:
            ss = np.where(XLA == jj[0, 0])[0]
            YL = Y
            XTL = np.c_[gs, XT[:, ii]]
            XTLQ = copy.deepcopy(XTL[ss, :])
            XTL = np.delete(XTL, ss, 0)
            XTLQ0 = copy.deepcopy(XT0[ss, :])
            XTL0 = np.delete(XT0, ss, 0)
            YL = np.delete(YL, ss, 0)
            x_train = np.c_[XTL0, XTL]
            x_test = np.c_[XTLQ0, XTLQ]
            beta = reg(x_train, YL)
            Y_cal_0 = cal(x_test, beta)
            YL_cal[ss, :] = Y_cal_0
        r2_0 = statistic.r2(YL_cal, Y, 1)
        aae = statistic.aae(YL_cal, Y, 1)
        aad = statistic.aad(YL_cal, Y, 1)
        rmse = statistic.rmse(YL_cal, Y, 1)
        return r2_0, aae, aad, rmse, YL_cal
