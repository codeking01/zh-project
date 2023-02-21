import copy
import warnings

import numpy as np
import pandas as pd

from common.StepwiseRegressionFunction import QSPR
# difine by our group
# former: from data import XT_SP_d_train, XT_SP_d_test, Y_train, Y_test, cd_n, nY_train, nY_test, XT0_train, XT0_test, cd_e
# from ML_JuneTask.Pre_data import X_train as XT_SP_d_train,X_test as XT_SP_d_test,y_train as Y_train,y_test as Y_test

def train_model_ad(train_cols=None,cd_n=None,cd_e=None,XT_SP_d_train=None,XT_SP_d_test=None,Y_train=None,Y_test=None):
    XT0_train=np.ones(np.shape(Y_train))
    XT0_test=np.ones(np.shape(Y_test))
    nY_train=np.shape(Y_train)[0]
    nY_test=np.shape(Y_test)[0]
    forward = QSPR.ForwardExternal
    status = QSPR.status
    from regress_method import regress_method as regr
    from regress_method import regress_name as rn

    warnings.filterwarnings("ignore")

    # =============================================================================
    # forward stepwise regression evaluated by external validation
    # ii: the selected index for independent variable in one models
    # II: ii for all models
    # bb: the parameters in one models
    # BB: bb for all models
    # result: the results for all models
    # n_max: the max number of independent variables selected in the model
    # n_ad: the number of  independent variables added one time
    # n_ad_t: the add time
    # n_de: the number of  independent variables deleted one time
    # n_ad_t_m: the max n_ad_t
    # combination: the combination of selected independent variables and  stepwise infromation
    #              it is a dictionary
    #              the key is f'{str(len(ii))}_{str(n_ad_t)}'
    #              the value is [n_de,n_ad,ii]
    # XT_train: independent variable of training set
    # XT_test: independent variable of test set
    # srf: the srf defined for stepwise regression
    #           ad_external: forward stepwise regression evaluated by external validation
    #           status: the status of the model
    #           evl: statistic parameter for evaluating model including 'r2,'aae', 'rmse', 'total'
    # =============================================================================
    headers = ['n', 'nY_test', 'nY_test', 'r2', 'aad', 'aae', 'r2_train', 'aad_train', 'aae_train', 'r2_test', 'aad_test',
               'aae_test', 'q2_train', 'aaeq_train', 'aadq_train', 'rmseq_train', 'q2', 'aaeq', 'aadq', 'rmseq']

    # =========================================================
    name_initial = f'result_ad_sp_{rn}'
    name_combination = f'combination_ad_sp_{rn}'
    # =========================================================

    ii = []
    II = []
    BB = []
    result = []
    n_max = train_cols
    n_ad = 1
    n_ad_t = 1
    n_ad_t_m = 20
    evl = 'total'
    XT_train = copy.deepcopy(XT_SP_d_train)
    XT_test = copy.deepcopy(XT_SP_d_test)
    combination = {}
    for i_all in range(max([len(ii), 1]) - 1, n_max):
        [ii, bb] = forward(regr, n_ad, n_ad_t, evl, Y_train, Y_test, ii, XT_train, XT_test, XT0_train, XT0_test)
        [r2, aae, aad, r2_train, aae_train, aad_train, r2_test, aae_test, aad_test] = status(regr, Y_train, Y_test,
                                                                                             XT_train, XT_test, ii,
                                                                                             XT0_train, XT0_test, bb)
        II.append(ii)
        BB.append(bb)
        for j in range(0, n_ad_t_m):
            combination[f'{str(i_all + 1)}_{str(j)}'] = np.mat(np.zeros((1, len(ii) + 2)))
        result.append(
            [i_all + 1, nY_train, nY_test, r2, aad, aae, r2_train, aad_train, aae_train, r2_test, aad_test, aae_test, 0, 0,
             0, 0, 0, 0, 0, 0])
        p_r = [i_all + 1, r2, aae, r2_train, aae_train, r2_test, aae_test]
        print(list(np.array(['%.4f' % i for i in p_r]).astype(float)))

    cd_n=cd_n
    cd_e=cd_e
    np.savez(f'{cd_n}{name_initial}.npz', II=II, result=result, BB=BB)
    np.save(f'{cd_n}{name_combination}.npy', combination)
    df = pd.DataFrame(result)
    df.to_excel(f'{cd_e}{name_initial}.xlsx', index=False, header=headers)
