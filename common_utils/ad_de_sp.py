import copy
import warnings

import numpy as np
import pandas as pd
from numpy import mat

from common.StepwiseRegressionFunction import QSPR
# difine by our group
from data import XT_SP_d_train, XT_SP_d_test, Y_train, Y_test, cd_n, cd_e, nY_train, nY_test, XT0_train, XT0_test

forward = QSPR.ForwardExternal
status = QSPR.status
backward = QSPR.BackwardExternal
from regress_method import regress_method as regr
from regress_method import regress_name as rn

warnings.filterwarnings("ignore")
# =============================================================================
# forward and backward stepwise regression evaluated by external validation
# ii: the selected index for independent variable in one models
# II: ii for all models
# bb: the parameters in one models
# BB: bb for all models
# result: the results for all models
# n_max: the max number of independent variables selected in the model
# n_ad: the number of  independent variables added one time
# n_ad_t: the add time
# n_ad_t_1: the min add time
# n_ad_t_e: the max add time
# n_de: the number of  independent variables deleted one time
# combination: the combination of selected independent variables and  stepwise infromation
#              it is a dictionary
#              the key is f'{str(len(ii))}_{str(n_ad_t)}'
#              the value is [n_de,n_ad,ii]
# XT_train: independent variable of training set
# XT_test: independent variable of test set
# srf: the srf defined for stepwise regression
#           ad_external: forward stepwise regression evaluated by external validation
#           status: the status of the model 
#           de_external: backward stepwise regression evaluated by external validation
# =============================================================================
headers = ['n', 'nY_test', 'nY_test', 'r2', 'aad', 'aae', 'r2_train', 'aad_train', 'aae_train', 'r2_test', 'aad_test',
           'aae_test', 'q2_train', 'aaeq_train', 'aadq_train', 'rmseq_train', 'q2', 'aaeq', 'aadq', 'rmseq']

name_model = f'result_sp_{rn}'
name_initial = f'result_ad_sp_{rn}'
name_combination = f'combination_ad_sp_{rn}'

beg_1 = 5
beg_e = 25
n_ad_t_1 = 6
n_ad_t_e = 15
n_ad = 1
n_max = 40
n_de = 3
evl = 'total'
XT_train = copy.deepcopy(XT_SP_d_train)
XT_test = copy.deepcopy(XT_SP_d_test)

for n_ad_t in range(n_ad_t_1, n_ad_t_e):
    for beg in range(beg_1, beg_e):
        print([n_ad_t, beg])
        # the initial index for independent variable in the model
        re_all = np.load(f'{cd_n}{name_initial}.npz', allow_pickle=True)
        II_tem = re_all['II']
        ii = II_tem[beg - 1]
        # the result of old model
        re_all = np.load(f'{cd_n}{name_model}.npz', allow_pickle=True)
        II = re_all['II']
        result = re_all['result']
        BB = re_all['BB']
        combination = np.load(f'{cd_n}{name_combination}.npy', allow_pickle=True).item()
        for i_all in range(len(copy.deepcopy(ii)) - 1, n_max):
            r2_ii_a = min(result[len(ii) - 1][6], result[len(ii) - 1][9])
            r2_min0 = -1
            n_break = 0
            r2_min = 0
            while r2_min > r2_min0:
                r2_min0 = r2_min
                [ii, bb] = forward(regr, n_ad, n_ad_t, evl, Y_train, Y_test, ii, XT_train, XT_test, XT0_train, XT0_test)
                [ii, bb] = backward(regr, n_de, n_ad_t * n_ad, evl, Y_train, Y_test, ii, XT_train, XT_test, XT0_train,
                                    XT0_test)
                [r2, aae, aad, r2_train, aae_train, aad_train, r2_test, aae_test, aad_test] = status(regr, Y_train,
                                                                                                     Y_test, XT_train,
                                                                                                     XT_test, ii,
                                                                                                     XT0_train,
                                                                                                     XT0_test, bb)
                r2_min = min(r2_train, r2_test)
                if r2_min > r2_min0:
                    ii_ = mat([n_de, n_ad] + np.sort(copy.deepcopy(ii)).tolist())
                    ii_0 = np.sum(abs(combination[str(i_all + 1) + '_' + str(n_ad_t)] - ii_), 1)
                    ii_0_w = len(np.where(ii_0 == 0)[0])
                    if ii_0_w == 0:
                        combination[str(i_all + 1) + '_' + str(n_ad_t)] = np.r_[
                            combination[str(i_all + 1) + '_' + str(n_ad_t)], ii_]
                    else:
                        n_break = 1
                        break
                    if r2_min > r2_ii_a:
                        r2_ii_a = r2_min
                        # the result of new model
                        II[i_all] = ii
                        BB[i_all] = bb
                        result[i_all, :] = [i_all + 1, nY_train, nY_test, r2, aad, aae, r2_train, aad_train, aae_train,
                                            r2_test, aad_test, aae_test, 0, 0, 0, 0, 0, 0, 0, 0]
                        p_r = [i_all + 1, r2, aae, r2_train, aae_train, r2_test, aae_test]
                        print(['%.4f' % i for i in p_r])
                        np.savez(f'{cd_n}{name_model}.npz', II=II, result=result, BB=BB)
                        np.save(f'{cd_n}{name_combination}.npy', combination)
                        df = pd.DataFrame(result)
                        df.to_excel(f'{cd_e}{name_model}.xlsx', index=False, header=headers)
                        II_tem[i_all] = ii
                else:
                    # add independent variables if  the model is not rivised
                    ii = II_tem[i_all]
                    [ii, bb] = forward(regr, 1, 1, evl, Y_train, Y_test, ii, XT_train, XT_test, XT0_train, XT0_test)
                    II_tem[i_all + 1] = ii
                    # stop stepwise regression if the combination is exist
            if n_break > 0:
                print('break')
                break
