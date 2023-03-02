"""
    -- coding: utf-8 --
    @Author: codeking
    @Data : 2022/6/18 23:55
    @File : PreDeal_Tools.py
"""
import copy
from copy import deepcopy
from math import ceil

# 所有的方法汇总
# 先判断缺失数据的的条件 这个以百分之70为基准
import numpy as np


def Del_deletion_data(dataValue, flag):
    """
    :param dataValue: 需要处理的数据
    :param flag: flag是1则删除列，如果是0则删除行
    :return:
    """
    dataValueCopy = deepcopy(dataValue)
    # 删除缺失值过多的列
    if flag == 1:
        del_cols = []
        # 先拿到元数据的长度
        baseLength = dataValueCopy.shape[0]
        # 这个 用来保存需要删除的列
        iterLength = dataValueCopy.shape[1]
        for i in range(iterLength):
            temp = []
            temp = np.where(np.isnan(dataValueCopy[:, i].astype(float)) == True)[0].tolist() + temp
            # print(temp)
            # 用集合去重
            c = set(temp)
            # 计算缺失的总行数
            lack_of_rows = len(c)
            if lack_of_rows > 0.3 * baseLength:
                del_cols.append(i)
                # print(dataValueCopy)
        # flag是1则删除列，如果是0则删除行
        final_data = np.delete(dataValueCopy, del_cols, axis=flag)
        # elif flag == 0:
        #     del_raws=[]+temp
        return final_data, del_cols
    # 删除含有缺失值的行
    elif flag == 0:
        try:
            # 筛选出含有缺失值的行，用set去重
            del_raws = set(np.where(np.isnan(dataValueCopy.astype(float)) == True)[0].tolist())
            del_raws = list(del_raws)
            final_data = np.delete(dataValueCopy, del_raws, axis=flag)
        except Exception as e:
            print(f"删除行失败：{e}")
        return final_data


# 将 数据比较一下:① 必须是两边都有的数据才可以进行预测或者建模，②数据量必须是超过70%
def Record_usable_cols(datavalue):
    # 先拿到元数据的长度
    baseLength = datavalue.shape[0]
    iterLength = datavalue.shape[1]
    # 这个 用来保存需要用列
    use_cols = []
    for i in range(iterLength):
        temp = []
        # 根据缺失值，将索引存到列表中
        temp = np.where(np.isnan(datavalue[:, i].astype(float)) == True)[0].tolist() + temp
        # print(temp)
        # 用集合去重
        c = set(temp)
        # 计算缺失的总行数
        lack_of_rows = len(c)
        if (lack_of_rows < 0.3 * baseLength):
            use_cols.append(i)
            # print(datavalue)
    return use_cols


# 将两个数据的能用的列合并 判断两边的数据是否都存在
# Original_Ydata_usableCols = Record_usable_cols(Original_Ydata)
# pred_Ydata_usableCols = Record_usable_cols(pred_Ydata)
# final_cols = list(set(Original_Ydata_usableCols + pred_Ydata_usableCols))
# 这个里面的内容可以用来建模与预测
# final_cols

# 更改Y_data的为-1 0 1 获取Y_data,然后获取5%的边界最小值Y_data_boundsMin，和95%的边界最大值Y_data_boundsMax
def Deal_sorted_Ydata(data):
    # 用来记录原顺序的数组，取反
    # order = np.argsort(-data)
    sort_data = copy.deepcopy(data)
    sort_data = sorted(sort_data)
    # sorted返回的是数组类型，这个地方需要转成ndarray
    sort_data = np.array(sort_data)
    data_length = sort_data.shape[0]
    # 根据5%为1，中间90%为0，后5%为-1
    # 这个地方要考虑临界条件，当处于临界条件的值，全部分配给两端
    bound_min_value = sort_data[int(ceil(0.05 * data_length))]
    bound_max_value = sort_data[int(ceil(0.95 * data_length))]
    # print('min长度', int(ceil(0.05 * data_length)))
    # print(bound_min_value, bound_max_value)
    min_data = np.where(data <= bound_min_value)
    # print('min_data', min_data)
    max_data = np.where(data >= bound_max_value)
    # print('max_data', max_data)
    normal_data = np.where((data > bound_min_value) & (data < bound_max_value))
    # print('normal_data', normal_data)
    # 统一进行替换为-1，1，0
    data[min_data] = -1
    data[max_data] = 1
    data[normal_data] = 0
    # data[0:int(data_length*0.05)]=1
    # data[int(data_length*0.05):int(data_length*0.95)]=0
    # data[int(data_length*0.95):]=-1
    # 恢复原来的顺序
    # recovery_arr = np.zeros_like(data)
    # for idx, num in enumerate(data):
    #     recovery_arr[order[idx]] = num
    return data, bound_min_value, bound_max_value


# 替换为-1，0，1的分类
# temp_data = Deal_sorted_Ydata(Y_data)
# Y_data = temp_data[0]
# Y_data_boundsMin = temp_data[1]
# Y_data_boundsMax = temp_data[2]

def check_string(data=None):
    """
    :param data: 一个列表
    :return: 是否存在字符串
    """
    for item in data:
        if type(item) == str:
            return True
    return False


# 处理带关键字的
def convert_to_float(current_train_col=None, current_pred_col=None):
    """
    :param current_train_col:
    :param current_pred_col:
    :return:current_train_col, current_pred_col, convert_flag(1成功，0失败)
    """
    try:
        current_train_col = [float(item) for item in current_train_col]
        current_pred_col = [float(item) for item in current_pred_col]
        return current_train_col, current_pred_col, 1
    except Exception as e:
        return current_train_col, current_pred_col, 0


def find_keyword(func):
    def wrapper(*args):
        # 获取列数
        all_cols = args[0].shape[1]
        # del_list 存储需要删除的列
        del_list = []
        for i in range(0, all_cols):
            current_original_col = args[0][:, i]
            current_verify_col = args[1][:, i]
            # 判断字符串是否存在
            flag = check_string(data=current_original_col) & check_string(data=current_verify_col)
            # 转化类型，存在字符串的，大多是字母，所以处理这个部分即可
            if flag == True:
                # 有的输入存在问题，需要先处理将所有都转化为数字，如果失败了在考虑去处理关键字
                args[0][:, i], args[1][:, i], convert_flag = convert_to_float(current_train_col=args[0][:, i],
                                                                              current_pred_col=args[1][:, i])
                # 如果转化失败则再处理
                if convert_flag == 0:
                    current_pred_col = args[1][:, i]
                    args[0][:, i], args[1][:, i], del_list = func(current_original_col.astype(str),
                                                                  current_pred_col.astype(str), del_list, i)
        args0 = np.delete(args[0], del_list, axis=1)
        args1 = np.delete(args[1], del_list, axis=1)
        return (args0, args1, del_list)

    return wrapper


@find_keyword
def convert_to_num(original_data=None, pred_data=None, del_list=None, index=None):
    """
    :param original_data: 建模数据
    :param pred_data: 预测数据
    :param del_list: 需要删除的列
    :param index: 当前索引
    :return: 将关键字转化成数字的两个结果,del_list删除的列
    """
    # 不要破坏原始数据
    original_data_copy = copy.deepcopy(original_data)
    pred_data_copy = copy.deepcopy(pred_data)
    # 取出每一列的唯一值
    unique_original_value = np.unique(original_data_copy)
    unique_pred_value = np.unique(pred_data_copy)
    # 取出并集,这个地方只需要判断是否关键字一致
    intersection = set(unique_original_value) | set(unique_pred_value)
    if len(intersection) == len(unique_original_value):
        # 如果 'nan'在里面,需要删除掉
        if 'nan' in intersection:
            intersection.remove('nan')
        # 这种才处理
        index = -1
        # 把关键字挨个处理
        for i in intersection:
            original_data_copy = np.where(original_data_copy == f"{i}", index, original_data_copy)
            pred_data_copy = np.where(pred_data_copy == f"{i}", index, pred_data_copy)
            index += 1
    else:
        # 删除这一列
        del_list.append(index)
    return original_data_copy, pred_data_copy, del_list


# 删除多余的列，并且返回处理好的数据已经需要删除的列
def delAndGetCols(deal_value):
    # 删除部分的缺失值过多的列
    temp_TwoToThree_Xdata = Del_deletion_data(deal_value, 1)
    base_temp_TwoToThree_Xdata = temp_TwoToThree_Xdata[0]
    # 获取需要删除的列
    del_cols = temp_TwoToThree_Xdata[1]
    return base_temp_TwoToThree_Xdata, del_cols


# 获取可用的列
def get_final_useablecols(Original_Yvalue, Pred_Yvalue):
    # 将两个数据的能用的列合并 判断两边的数据是否都存在
    Original_Ydata_usableCols = Record_usable_cols(Original_Yvalue)
    pred_Ydata_usableCols = Record_usable_cols(Pred_Yvalue)
    # 取出交集
    # list(set(a).intersection(b))
    final_cols = list(set(Original_Ydata_usableCols).intersection(pred_Ydata_usableCols))
    # final_cols = list(set(Original_Ydata_usableCols + pred_Ydata_usableCols))
    # 这个里面的内容可以用来建模与预测(可用的列)
    return final_cols


# 获取所有的Ydata(从final_cols中获取)
def all_ydata(final_cols, original_yvalue, pred_yvalue):
    # 取出Y_data 元数据
    # 用来存储所有的Y_data
    Original_YdataList = []
    Pred_YdataList = []
    for index in final_cols:
        Original_YdataList.append(original_yvalue[:, index])
        Pred_YdataList.append(pred_yvalue[:, index])
    return Original_YdataList, Pred_YdataList


# 根据传入的excel对象，将数据表合并
def get_merge_tabledata(excel_data=None):
    """
    todo 需要考虑如果行号改变的时候如何固定下各表的数据
    在处理某些特殊字段的时候，如果建模的数据有A,B,C,D，但是预测的数据只有B,C两类的话，那么会出现字母转成数字不匹配的现象，
    假如建模的变成{0,1,2,3},那么预测的数据就要变成{1，2}，并且这个数据是动态的（上次数据的表和这次的不一样），
    那么每次都需要动态获取，我有个想法，就是将这种数据直接统一化不作为一个变量了（比如都设置成1）,这个地方不考虑,直接排除
    :param excel_data: 传入的excel表的内容
    :return:
    """
    # 这个数据是公共部分 需要转化为数字
    common_head = excel_data[3:, 4:6]
    # 替换的方法 取出第1列 这个是需要动态去处理的,不能写死
    # original_data,pred_data = convert_to_num(common_head, common_head)
    # common_head[:, 0] = convert_to_num(common_head[:, 0])
    # common_head[:, 1] = convert_to_num(common_head[:, 1])
    # 处理表1和表2
    TABLE_ONE = excel_data[3:, 6:18]
    #   自动处理关键字内容(汉字等)  如果是汉字,这个地方必须转成字符串
    # TABLE_ONE[:, 12] = convert_to_num(TABLE_ONE[:, 12].astype(str))
    TABLE_TWO = excel_data[3:, 18:21]
    # 处理表3
    TABLE_THREE = excel_data[3:, 27:30]
    # 加上工序二的输出
    # TABLE_THREE = excel_data[3:, 21:30]

    # 这个地方必须转成字符串   处理掉汉字
    # TABLE_THREE[:, -1] = convert_to_num(TABLE_THREE[:, -1].astype(str))
    # 处理表4 工序四没有输出out,所有直接和工序五合在一起
    TABLE_FOUR = excel_data[3:, 34:49]
    # 加上工序三的输出
    # TABLE_FOUR = excel_data[3:, 30:49]

    # TABLE_FOUR[:, -1] = convert_to_num(TABLE_FOUR[:, -1].astype(str))
    # 处理表5 处理的是工序六
    TABLE_FIVE = excel_data[3:, 57:70]
    #   加上工序五的输出
    # TABLE_FIVE = excel_data[3:, 49:70]

    # TABLE_FIVE[:, -1] = convert_to_num(TABLE_FIVE[:, -1].astype(str))
    # 处理表6 处理的是工序七
    TABLE_SIX = excel_data[3:, 83:87]
    # 加上工序六的输出
    # TABLE_SIX = excel_data[3:, 70:87]

    mergeTwo_data = np.column_stack((common_head, TABLE_ONE, TABLE_TWO))
    mergeThree_data = np.column_stack((mergeTwo_data, TABLE_THREE))
    mergeFour_data = np.column_stack((mergeThree_data, TABLE_FOUR))
    mergeFive_data = np.column_stack((mergeFour_data, TABLE_FIVE))
    mergeSix_data = np.column_stack((mergeFive_data, TABLE_SIX))
    return mergeTwo_data, mergeThree_data, mergeFour_data, mergeFive_data, mergeSix_data


# 根据传入的excel对象获取Y数据
def get_former_Ydata(excel_data):
    # 处理所有建模的Y_data 处理工序二的数据
    Merge_TableTwoYdata = excel_data[3:, 21:27]
    # 处理工序三的数据
    Merge_TableThreeYdata = excel_data[3:, 30:34]
    # 没有Original_TableFourYdata 处理工序五的数据
    Merge_TableFiveYdata = excel_data[3:, 49:57]
    # 处理工序六的数据
    Merge_TableSixYdata = excel_data[3:, 70:83]
    # 处理工序七的数据
    Merge_TableSevenYdata = excel_data[3:, 87:106]
    return Merge_TableTwoYdata, Merge_TableThreeYdata, Merge_TableFiveYdata, Merge_TableSixYdata, Merge_TableSevenYdata


# 根据Y_data_boundsMin和Y_data_boundsMax处理Y_data
def deal_verify_ydata(Verify_Ydata, Y_data_boundsMin, Y_data_boundsMax):
    # 将pred_Ydata 替换成-1，0，1的分类
    min_data = np.where(Verify_Ydata <= Y_data_boundsMin)
    # print('min_data', min_data)
    max_data = np.where(Verify_Ydata >= Y_data_boundsMax)
    # print('max_data', max_data)
    normal_data = np.where((Verify_Ydata > Y_data_boundsMin) & (Verify_Ydata < Y_data_boundsMax))
    # print('normal_data', normal_data)
    # 统一进行替换为-1，1，0
    Verify_Ydata[min_data] = -1
    Verify_Ydata[max_data] = 1
    Verify_Ydata[normal_data] = 0
    return Verify_Ydata
