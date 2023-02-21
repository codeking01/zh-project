# author: code_king
# time: 2023/2/20 17:02
# file: examples.py
def del_nan():
    import numpy as np
    # 删除nan
    myArray = np.array([1, 2, 3, np.nan, np.nan, 4, 5, 6, np.nan, 7, 8, 9, np.nan])
    output1 = myArray[np.logical_not(np.isnan(myArray))] # Line 1
    output2 = myArray[~np.isnan(myArray)] # Line 2
