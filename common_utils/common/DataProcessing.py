import numpy as np
import copy


class GroupContribution:
    # =============================================================================
    # 判断相关性: x=a*y
    # 存在x数值为常数的可能，所以不能用r2
    # x/(x+y) 为常数表明：x=a*y
    # 输出符合这种情况的索引
    # xy 为字典格式数据
    # =============================================================================
    def CorrelationDic(xy):
        k = list(xy.keys())
        SS_w_del = []
        l_k = len(k)
        for i in range(l_k - 1):
            for j in range(i + 1, l_k):
                X1 = xy[k[i]]
                X2 = xy[k[j]]
                x_x = X1 / (X1 + X2)
                xx_w = np.where(~np.isnan(x_x))[0]
                if len(xx_w) > 0:
                    x_x = x_x[xx_w, :]
                    if max(x_x)[0, 0] == min(x_x)[0, 0]:
                        SS_w_del.extend(xx_w.tolist())
        SS_w = np.arange(len(X1))
        SS_w = np.delete(SS_w, SS_w_del, 0)
        return np.array(SS_w)

    # =============================================================================
    # 判断相关性: x=a*y
    # 存在x数值为常数的可能，所以不能用r2
    # x/(x+y) 为常数表明：x=a*y
    # 输出符合这种情况的索引
    # xy 为矩阵格式数据
    # =============================================================================
    def CorrelationMat(xy):
        SS_w_del = []
        l_k = np.shape(xy)[1]
        for i in range(l_k - 1):
            for j in range(i + 1, l_k):
                X1 = xy[:, i]
                X2 = xy[:, j]
                x_x = X1 / (X1 + X2)
                xx_w = np.where(~np.isnan(x_x))[0]
                if len(xx_w) > 0:
                    x_x = x_x[xx_w, :]
                    if max(x_x)[0, 0] == min(x_x)[0, 0]:
                        SS_w_del.extend(xx_w.tolist())
        SS_w = np.arange(len(X1))
        SS_w = np.delete(SS_w, SS_w_del, 0)
        return np.array(SS_w)

    # =============================================================================
    # 删掉基团出现次数少于6的物质
    # =============================================================================
    def Group6Delete(DataDic, DataMat, SS_w):
        S_key = list(DataDic['Group_N'].keys())
        ll = len(S_key)
        ll0 = len(S_key) + 1
        while ll < ll0:
            ll0 = copy.deepcopy(ll)
            ww = []
            for i in S_key:
                w_0 = np.where(DataDic['Group_N'][i] != 0)[0]
                l_0 = len(w_0)
                if l_0 < 6:
                    ww = ww + list(w_0)
                    for j in DataDic.values():
                        del j[i]
            ww_ = list(set(ww))
            S_key = list(DataDic['Group_N'].keys())
            for i in DataDic.keys():
                for j in S_key:
                    DataDic[i][j] = np.delete(DataDic[i][j], ww_, 0)
            for i in DataMat.keys():
                DataMat[i] = np.delete(DataMat[i], ww_, 0)
            SS_w = np.delete(SS_w, ww_, 0)
            ll = len(S_key)
        return DataDic, DataMat, SS_w

    # =============================================================================
    # 根据索引删掉字典和矩阵数据
    # DataDic是字典数据
    # DataMat是矩阵数据
    # =============================================================================
    def DeleteByIndex(DataDic, DataMat, SS_w):
        for i in DataDic.keys():
            for j in DataDic[i]:
                DataDic[i][j] = DataDic[i][j][SS_w, :]
        for i in DataMat.keys():
            DataMat[i] = DataMat[i][SS_w, :]
        return DataDic, DataMat

    # =============================================================================
    # 定位因子组合：单个基团数据合并成矩阵
    # =============================================================================
    def PositionMerge(Position):
        Position = copy.deepcopy(Position)
        S_key = list(Position.keys())
        L_r = np.shape(Position[S_key[0]])[1]
        L = len(S_key)
        pf = {}
        for i in range(0, L_r):
            pf[i] = np.mat(Position[S_key[0]][:, i]).T
            for j in range(1, L):
                pf[i] = np.c_[pf[i], np.mat(Position[S_key[j]][:, i]).T]
        pf_all = pf[0]
        for i in range(1, L_r):
            pf_all = np.c_[pf_all, pf[i]]
        return pf_all
        # =============================================================================
    # =============================================================================
    # 定位因子组合：单个基团数据合并成矩阵
    # =============================================================================
    def PositionMergePredict(Position,g_name_s):
        Position = copy.deepcopy(Position)
        S_key = list(Position.keys())
        L_r = np.shape(Position[S_key[0]])[1]
        pf = {}
        l_g=len(g_name_s)    
        for i in range(0, L_r):
            pf[i]=np.mat(np.zeros((1,l_g)))
            for j in S_key:
                i_g=g_name_s.index(j)
                pf[i][0,i_g] = Position[j][0, i]
        pf_all = pf[0]
        for i in range(1, L_r):
            pf_all = np.c_[pf_all, pf[i]]
        return pf_all
    # 基团组合：单个基团数据合并成矩阵
    # =============================================================================
    def GroupMerge(Group):
        Group = copy.deepcopy(Group)
        key = list(Group.keys())
        L_r = np.shape(Group[key[0]])[1]
        group = {}
        for i in range(0, L_r):
            k = 0
            for j in key:
                if k == 0:
                    group[i] = Group[j][:, i]
                    k += 1
                else:
                    group[i] = np.c_[group[i], Group[j][:, i]]
        return group

    # =============================================================================    
    # 基团组合：单个基团数据合并成矩阵
    # =============================================================================
    def GroupMergePredict(Group,g_name_s):
        Group = copy.deepcopy(Group)
        key = list(Group.keys())
        L_r = np.shape(Group[key[0]])[1]
        group = {}
        l_g=len(g_name_s)    
        for i in range(0, L_r):
            group[i]=np.mat(np.zeros((1,l_g)))
            for j in key:
                i_g=g_name_s.index(j)
                group[i][:,i_g] = Group[j][:, i]
        return group

    # =============================================================================
    # 识别训练集和测试集
    # 每个基团在训练集中至少出现5次
    # DataMat是矩阵数据
    # =============================================================================
    def TrainTest(g_n, Range):
        g_n = copy.deepcopy(g_n)
        ly = np.shape(g_n)[0]
        g_n[g_n != 0] = 1
        L = 0
        t_t = []
        for j in Range:
            jj = copy.deepcopy(j)
            while L < ly / 5 and jj + 5 < ly - 1:
                t_t_ = copy.deepcopy(t_t)
                t_t_.append(jj)
                g_n_ = np.delete(g_n, t_t_, 0)
                g_n_s = np.sum(g_n_, 0)
                if np.min(g_n_s) > 4:
                    t_t.append(jj)
                jj = jj + 5
                L = len(t_t)
        return t_t


class common:
    # =============================================================================
    # 划分训练集和测试集：矩阵数据
    # =============================================================================
    def TrainTestMat(data, TT):
        data_train = np.delete(data, TT, 0)
        data_test = data[TT, :]
        return data_train, data_test

    # =============================================================================
    # 划分训练集和测试集：字典数据
    # =============================================================================
    def TrainTestDic(data, TT):
        data_train, data_test = {}, {}
        for i in data.keys():
            data_train[i] = np.delete(data[i], TT, 0)
            data_test[i] = data[i][TT, :]
        return data_train, data_test

    # =============================================================================
    # 识别物质描述符是否都为0：x为字典数据
    # =============================================================================
    def X0Dic(x, L):
        x_0_i = np.zeros((1, L))
        S_key = list(x.keys())
        for i in S_key:
            x_0_i += np.sum(x[i], 1)
        return np.mat(x_0_i).T

    # =============================================================================
    # 识别物质描述符是否都为0：x为字典数据
    # =============================================================================
    def X0Mat(x, L):
        x_0_i = np.zeros((1, L))
        S_key = list(x.keys())
        for i in S_key:
            x_0_i += np.sum(x[i], 1)
        return np.mat(x_0_i).T

    def test_select(index,XLA):
        XLA_lst=XLA.tolist()
        lst_1=[]
        lst_2=[]
        for i in XLA_lst:
            if i[0] not in lst_1:
                lst_1.append(i[0])
                XLA_L=np.shape(np.where(XLA==i[0])[0])[0]
                lst_2.append(XLA_L)
                
        XLA_L_index=np.argsort(lst_2)       #   从小到大
        lst_3=XLA_L_index.tolist()
        test_lst=[]
        
        for i in range(len(lst_3)):
            if (i % 10 == index[0]) or (i % 10 == index[1]):
                test_lst.append(lst_1[lst_3[i]])
        test_lst.sort()
        
        # test_lst=np.array(test_lst)
        TT=[]
        for i in range(len(test_lst)):
            TT_1=list(np.where(XLA==test_lst[i])[0])
            TT=TT+TT_1
        
        TT=np.array(TT)
        return TT
