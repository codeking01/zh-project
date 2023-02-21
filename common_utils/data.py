import os
import numpy as np
import pandas as pd
from numpy import mat
import copy
import warnings
#
warnings.filterwarnings("ignore")
cd_o=os.getcwd()
cd_p=os.path.dirname(cd_o)
p_name='Antoine'
cd_s=f'{cd_p}/structure/'
cd_e=f'{cd_p}/data_excel/'
cd_n=f'{cd_p}/data_npy/'
cd_fig=f'{cd_p}/figure/'
from common.DataProcessing import common as dpc
data_a= pd.read_excel(f'{cd_e}{p_name}.xlsx',sheet_name='data')
L = np.shape(data_a)[0]
r_name = mat(data_a['CAS']).T
Y = mat(data_a['value']).T
ss = mat(data_a['select']).T
Tb = mat(data_a['Tb']).T
XLA = mat(data_a['XLA']).T
T = mat(data_a['Temperature (K)']).T
# Y是 训练的
Y = np.log(Y / 1000)

# ii_sp=np.load(f'{cd_n}ii_left_sp.npy',allow_pickle=True)
# # ii_sq=np.load(f'{cd_n}ii_left_sq.npy',allow_pickle=True)
#
# NISP=np.load(f'{cd_n}NISP.npz',allow_pickle=True)
# X_SP=mat(NISP['NISP'])
# name_NISP=mat(NISP['name_NISP'])
#
# # NISQ=np.load(f'{cd_n}NISQ.npz',allow_pickle=True)
# # X_SQ=mat(NISQ['NISQ'])
# # name_NISQ=mat(NISQ['name_NISQ'])
#
# IFS=np.load(f'{cd_n}IFS.npy',allow_pickle=True)
# IFS=mat(IFS)
# #IFD=np.load(f'{cd_n}IFD.npy',allow_pickle=True)
# #IFD=mat(IFD)
#
# # x_s_0=np.sum(X_SQ,1)
# # x_d_0=np.sum(X_SQ,1)
# # X_SP=X_SP[:,ii_sp]
# # X_SQ=X_SQ[:,ii_sq]
#
# name_NISP_s=name_NISP[:,ii_sp].tolist()[0]
# name_NISP_s_0=copy.deepcopy(name_NISP_s)
# name_NISP_s_1=copy.deepcopy(name_NISP_s)
# name_NISP_s_2=copy.deepcopy(name_NISP_s)
# name_NISP_s_3=copy.deepcopy(name_NISP_s)
# name_NISP_s_4=copy.deepcopy(name_NISP_s)
# for i in range(0,len(ii_sp)) :
#     name_NISP_s_1[i]=name_NISP_s_1[i]+'/IFS[:,1]'
#     name_NISP_s_2[i]=name_NISP_s_2[i]+'/np.sqrt(IFS[:,2]'
#     name_NISP_s_3[i]=name_NISP_s_3[i]+'/np.sqrt(IFS[:,3]'
#     name_NISP_s_4[i]=name_NISP_s_4[i]+'/np.sqrt(IFS[:,4]'
#
# # name_NISQ_s=name_NISQ[:,ii_sq].tolist()[0]
# # name_NISQ_s_0=copy.deepcopy(name_NISQ_s)
# # name_NISQ_s_1=copy.deepcopy(name_NISQ_s)
# # name_NISQ_s_2=copy.deepcopy(name_NISQ_s)
# # name_NISQ_s_3=copy.deepcopy(name_NISQ_s)
# # name_NISQ_s_4=copy.deepcopy(name_NISQ_s)
# # for i in range(0,len(ii_sq)) :
# #     name_NISQ_s_1[i]=name_NISQ_s_1[i]+'/IFS[:,1]'
# #     name_NISQ_s_2[i]=name_NISQ_s_2[i]+'/np.sqrt(IFS[:,2]'
# #     name_NISQ_s_3[i]=name_NISQ_s_3[i]+'/np.sqrt(IFS[:,3]'
# #     name_NISQ_s_4[i]=name_NISQ_s_4[i]+'/np.sqrt(IFS[:,4]'
#
# name_NISP_s_all=name_NISP_s_0+name_NISP_s_1+name_NISP_s_2+name_NISP_s_3+name_NISP_s_4
# # name_NISQ_s_all=name_NISQ_s_0+name_NISQ_s_1+name_NISQ_s_2+name_NISQ_s_3+name_NISQ_s_4
# XT_SP_d=np.c_[copy.deepcopy(X_SP),copy.deepcopy(X_SP)/IFS[:,1],copy.deepcopy(X_SP)/np.sqrt(IFS[:,2]),copy.deepcopy(X_SP)/np.sqrt(IFS[:,1]/IFS[:,3]),copy.deepcopy(X_SP)/np.sqrt(IFS[:,4])]
# # XT_SQ_d=np.c_[copy.deepcopy(X_SQ),copy.deepcopy(X_SQ)/IFS[:,1],copy.deepcopy(X_SQ)/np.sqrt(IFS[:,2]),copy.deepcopy(X_SQ)/np.sqrt(IFS[:,1]/IFS[:,3]),copy.deepcopy(X_SQ)/np.sqrt(IFS[:,4])]
# # XT_SPSQ_d=np.c_[copy.deepcopy(XT_SP_d),copy.deepcopy(XT_SQ_d)]
# na=copy.deepcopy(IFS[:,1])
# # SS_w=np.where((IFS[:,3]>2)&(ss==1)&(x_s_0!=0)&(x_d_0!=0)&(~np.isnan(Y)))[0]
SS_w = np.where((IFS[:, 3] > 2) & (ss == 1) & (~np.isnan(Y)))[0]
r_name = r_name[SS_w, :]
Y = Y[SS_w, :]
T = T[SS_w, :]
Tb = Tb[SS_w, :]
XT_SP_d = XT_SP_d[SS_w, :]
# XT_SQ_d=XT_SQ_d[SS_w,:]
X_SP = X_SP[SS_w, :]
# X_SQ=X_SQ[SS_w,:]
IFS = IFS[SS_w, :]
XLA = XLA[SS_w, :]
na = na[SS_w, :]

C = -25
C_T_R = -1 / (T + C)

XT_SP_d1 = np.multiply(XT_SP_d, C_T_R)
XT_SP_d = np.c_[XT_SP_d, XT_SP_d1]

TT = dpc.test_select([1, 6], XLA)

Y_train, Y_test = dpc.TrainTestMat(Y, TT)
XLA_train, XLA_test = dpc.TrainTestMat(XLA, TT)
XT_SP_d_train, XT_SP_d_test = dpc.TrainTestMat(XT_SP_d, TT)

# XT_SPSQ_d_train,XT_SPSQ_d_test=dpc.TrainTestMat(XT_SPSQ_d,TT)
X_SP_train, X_SP_test = dpc.TrainTestMat(X_SP, TT)
# X_SQ_train,X_SQ_test=dpc.TrainTestMat(X_SQ,TT)
IFS_train, IFS_test = dpc.TrainTestMat(IFS, TT)
C_T_R_train, C_T_R_test = dpc.TrainTestMat(C_T_R, TT)


nY_test=np.shape(Y_test)[0]
nY_train=np.shape(Y_train)[0]

XT0=np.ones(np.shape(Y))
XT0_train=np.ones(np.shape(Y_train))
XT0_test=np.ones(np.shape(Y_test))