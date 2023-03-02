import os

import matplotlib.pyplot as plt
import numpy as np


# x_left=np.array([1,20,3,1,20,3])
# x_right=np.array([1,26,3,1,20,3])
#
# X_left=x_left
# X_right=x_right
# names="sss"

def Freq_cal(X=None):
    x_r = []
    Num = []
    X = np.array(sorted(X))
    if X.dtype == "<U2":
        X = X.tolist()
        X = [int(i) for i in X]
        X = np.array(X)
    X_mean = np.mean(X)
    ss_w = np.where(X <= X_mean)[0].shape[0]
    for i in range(ss_w):
        Num.append(np.shape(np.where(X <= X[i]))[1])
        x_r.append(X[i])
    for i in range(ss_w, X.shape[0]):
        Num.append(np.shape(np.where(X >= X[i]))[1])
        x_r.append(X[i])
    return x_r, Num


def del_nan(deal_value=None):
    """
    :param deal_value: 需要处理的列表
    :return:
    """
    deal_value = list(deal_value)
    # 删除所有的nan,转成列表
    # 将这个转化为字符串，然后删除 'nan'，然后再将类型转化为 float
    deal_value_list = [str(item) for item in deal_value]
    # 删除列表中的
    for i in range(len(deal_value_list) - 1, -1, -1):
        if (deal_value_list[i] == "nan") or (deal_value_list[i] == 'nan'):
            deal_value_list.remove(deal_value_list[i])
    try:
        deal_value_list = [float(i) for i in deal_value_list]
        return np.array(deal_value_list)
    except Exception as e:
        print(f"存在字母，不画图了，失败原因：{e}")


def write_images(data_left, data_right, names=None, score=None):
    """
    :param data_left: 第一个表的 单列
    :param data_right: 第二个表 单列
    :param names: 图的名字
    :param score: 分数
    :return:
    """
    # todo 这个地方只考虑了一级目录如果是多级目录，可用递归去创建，不写了
    if "/" in names:
        dir_name = names.split("/")[0]
        if not os.path.exists(f"{dir_name}"):
            os.makedirs(name=f"{dir_name}", exist_ok=True)
    try:
        data_left = del_nan(deal_value=data_left)
        data_right = del_nan(deal_value=data_right)

        X_left_two, Num_left = Freq_cal(data_left)
        X_right_one, Num_right = Freq_cal(data_right)
        lim_x_min = np.min(X_left_two + X_right_one)
        lim_x_max = np.max(X_left_two + X_right_one)
        lim_x = lim_x_max - lim_x_min
        lim_y_min = np.min(Num_left + Num_right)
        lim_y_max = np.max(Num_left + Num_right)
        lim_y = lim_y_max - lim_y_min

        plt.figure(figsize=(10, 10), dpi=150)
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 20,
                }
        plt.xlim((lim_x_min - 0.1 * lim_x, lim_x_max + 0.1 * lim_x))
        plt.ylim((lim_y_min - 0.1 * lim_y, lim_y_max + 0.1 * lim_y))
        plt.plot(X_left_two, Num_left, "o-", color='green', lw=3, label='models_build')
        plt.plot(X_right_one, Num_right, "o-", color='red', lw=3, label='models_pred')
        x_tick = np.linspace(lim_x_min, lim_x_max, 7)
        x_label = np.round(x_tick, 2)
        y_tick = np.linspace(lim_y_min, lim_y_max, 7)
        y_label = [int(i) for i in y_tick]
        plt.xticks(x_tick, x_label, size=20)
        plt.yticks(y_tick, y_label, size=20)
        if score != None:
            score = "%.3f" % score
            plt.text(lim_x_min, lim_y_max, f'Accuracy={score}', fontdict=font)
        plt.legend(loc='upper right', prop=font, labelspacing=1, frameon=False)
        plt.xlabel('Values', fontdict={'size': 30})
        plt.ylabel('Nums', fontdict={'size': 30})
        plt.savefig(f'{names}', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"画图出错，原因{e}")