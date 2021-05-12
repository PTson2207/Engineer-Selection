import pandas as pd
import numpy as np
#from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif,chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

def constant_feature_detect(data, threshold=0.98):
    """
    phát hiện các tính năng hiển thị cùng giá trị cho phần lớn/tất cả các quann sát
    """
    data_copy = data.copy(deep=True)
    quasi_constant_feature = []
    for fea in data_copy.columns:
        predominant = (data_copy[fea].value_counts()/np.float(len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append(fea)
    print(len(quasi_constant_feature), ' biến được tìm thấy gần như không đổi')
    return quasi_constant_feature

def corr_feature_detect(data, threshold=0.8):
    """
    Lấy các feature có độ tương quan cao
    """
    corrmat = data.corr()
    corrmat = corrmat.abs().unstack() # gia tri tuyet doi
    corrmat = corrmat.sort_values(ascending=True)
    corrmat = corrmat[corrmat >= threshold]
    corrmat = corrmat[corrmat < 1]
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'correlation']

    grouped_feature_ls = []
    correlated_groups = []

    for i in corrmat.feature1.unique():
        if i not in grouped_feature_ls:
            correlated_block = corrmat[corrmat.feature1 == i]
            grouped_feature_ls = grouped_feature_ls + list(correlated_block.feature2.unique()) + [i]
            correlated_groups.append(correlated_block)
    
    return correlated_groups