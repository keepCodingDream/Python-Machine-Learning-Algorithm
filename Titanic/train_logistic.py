# coding:UTF-8

"""
  ***强行使用线性回归拟合非线性数据，导致训练完全不着边际，所以需要选择好的模型***
  @Date 2017-11-11
  @author tracy
"""

import csv

import numpy as np

from LogisticRegression.lr_train import LogisticRegression


def loadData(file_name, label_index):
    """
    :param file_name: 文件名称
    :param label_index: 标签的坐标
    :return: 特征名称,原始数据(包含标签,矩阵),标签
    """
    workbook = list(csv.reader(open(file_name, 'rb')))
    data = []
    feature = workbook[0]
    for rol in range(1, len(workbook)):
        data.append(workbook[rol])
    data_matrix = np.mat(data)
    label = data_matrix[:, label_index]
    label_list = label.tolist()
    result = [float(item[0]) for item in label_list]
    return feature, data_matrix, result


if __name__ == "__main__":
    feature, data_matrix, label = loadData("train.csv", label_index=1)
    lr_train = LogisticRegression()
    choose_feature_list = ["Sex", "Age", "Fare"]
    data_list = []
    for i in range(len(feature)):
        if feature[i] in choose_feature_list:
            item_list = data_matrix[:, i].tolist()
            if "Sex" == feature[i]:
                sex_list = []
                for j in item_list:
                    if "female" == j[0]:
                        sex_list.append(0)
                    else:
                        sex_list.append(1)
                data_list.append(sex_list)
            elif "Age" == feature[i]:
                # age 已平均年龄补全缺失
                age_list = []
                for j in range(0, len(item_list)):
                    if "" == item_list[j][0]:
                        age_list.append(30)
                    else:
                        age_list.append(float(item_list[j][0]))
                data_list.append(age_list)
            else:
                else_list = []
                for j in item_list:
                    else_list.append(float(j[0]))
                data_list.append(else_list)

    w = lr_train.lr_train_bgd(np.mat(data_list).T, np.mat(label).T, 1000, 0.01)
    print w
