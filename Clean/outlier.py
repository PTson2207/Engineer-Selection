import pandas as pd
import numpy as np

def outlier_detect_arbitrary(data, col, uper_fence, lower_fence):
    """
    xác định các giá trị ngoại lai dựa trên các ranh giới tùy ý được chuyển đến hàm.
    """
    para = (uper_fence, lower_fence)
    tmp = pd.concat([data[col]>uper_fence, data[col]<lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    print("Number of oulier dectected:", outlier_index.value_counts()[1])
    print("Proportion of outlier detected:", outlier_index.value_counts()[1]/len(outlier_index))

    return outlier_index, para

def outlier_detect_IQR(data, col, threshold=3):

    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
    Upper_fence = data[col].quantile(0.75) - (IQR * threshold)
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col]>Upper_fence, data[col]<Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    print("Number of oulier dectected:", outlier_index.value_counts()[1])
    print("Proportion of outlier detected:", outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index, para


def outlier_detect_mean_std(data, col, threshold=3):
    Upper_fence = data[col].mean() + threshold * data[col].std()
    Lower_fence = data[col].mean() - threshold * data[col].std()
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col]>Upper_fence, data[col]<Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    print("Number of oulier dectected:", outlier_index.value_counts()[1])
    print("Proportion of outlier detected:", outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index, para

def outlier_detect_MAD(data, col, threshold=3.5):
    median = data[col].median()
    median_absolution_devaiation = np.median([np.abs(y - median) for y in data[col]])
    modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolution_devaiation for y in data[col]])
    outlier_index = np.abs(modified_z_scores) > threshold
    print("Num of outlier deteced:", outlier_index.value_counts()[1])
    print("Proportion of outlier detected:", outlier_index.value_counts()[1] / len(outlier_index))
    return outlier_index

def impute_outlier_with_arbitrary(data, outlier_index, value, col=[]):
    """
    đưa ra các giá trị ngoại lệ với giá trị tùy ý
    """
    data_copy = data.copy(deep=True)
    for i in col:
        data_copy.loc[outlier_index, i] = value
    return data_copy