# coding:UTF-8

"""
  @Date 2017-11-11
  @author tracy
"""

import csv
import logging
import logging.config
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
logging.config.fileConfig("logger.conf")
log = logging.getLogger("example01")

def readData(fileName):
    """
    :param fileName: 需要打开的文件名字
    :return: {key1:[],key2:[],……}以这种形式的字典
    attr_list存放所有特征名称
    """
    result = {}
    with open(fileName, 'rb') as f:
        rows = csv.reader(f)
        for row in rows:
            if result.has_key('attr_list'):
                for i in range(len(result['attr_list'])):
                    key = result['attr_list'][i]
                    if not result.has_key(key):
                        result[key] = []
                    result[key].append(row[i])
            else:
                result['attr_list'] = row
    return result


def writeData(fileName, data):
    csvFile = open(fileName, 'w')
    writer = csv.writer(csvFile)
    n = len(data)
    for i in range(n):
        writer.writerow(data[i])
    csvFile.close()


def convertData(dataList):
    hashTable = {}
    count = 0.0
    for i in range(len(dataList)):
        if not hashTable.has_key(dataList[i]):
            hashTable[dataList[i]] = count
            count += 1
        dataList[i] = hashTable[dataList[i]]


def convertValueData(dataList):
    """
    以平均数填充缺失值
    :param dataList: 入参数据
    :return:补充后的数据
    """
    sumValue = 0.0
    count = 0
    for i in range(len(dataList)):
        if dataList[i] == "":
            continue
        sumValue += float(dataList[i])
        count += 1
        dataList[i] = float(dataList[i])
    avg = sumValue / count
    for i in range(len(dataList)):
        if dataList[i] == "":
            dataList[i] = avg


def dataPredeal(data):
    convertValueData(data["Age"])
    convertData(data["Fare"])
    convertData(data["Pclass"])
    convertData(data["Sex"])
    convertData(data["SibSp"])
    convertData(data["Parch"])
    convertData(data["Embarked"])


def getX(data, ignores=None):
    """
    将所有不在ignores字典里的key添加到x二维数组内
    :param ignores: 标签中忽略的特征
    :param data: {key1:[],key2:[]……}
    :return:[[a1,b1,c1,d1……][a2,b2,c2,d2……]……],{feature1:1,feature2:1}
    """
    x = []
    if ignores is None:
        ignores = {"PassengerId": 1, "Survived": 1, "Name": 1, "Ticket": 1, "Cabin": 1, "Fare": 1, "Embarked": 1}
    selected_features = {}
    for i in range(len(data["PassengerId"])):
        x.append([])
        for j in range(len(data["attr_list"])):
            key = data["attr_list"][j]
            if not ignores.has_key(key):
                selected_features[key] = 1
                if "Age" == key:
                    data[key][i] /= 10
                x[i].append(int(data[key][i]))
    return x, selected_features


def getLabel(data):
    label = []
    for i in range(len(data["PassengerId"])):
        label.append(int(data["Survived"][i]))
    return label


def calResult(x, label, input_x):
    """
    svm 训练
    :param x:训练数据
    :param label: 数据标签
    :param input_x:测试数据
    :return: 决策结果
    """
    svmcal = svm.SVC(kernel='linear').fit(x, label)
    return svmcal.predict(input_x)


def performance(predict_result, real_result):
    fpr, tpr, thresholds = roc_curve(real_result, predict_result, pos_label=1)
    print(fpr)
    print(tpr)
    print(thresholds)
    print auc(fpr, tpr)


if __name__ == "__main__":
    dataSet = pd.read_csv('train.csv')
    rows = dataSet.head(1)
    for i in rows:
        # Age 做单独分析
        if i not in ('PassengerId', 'Survived', 'Name', 'Cabin', 'Ticket', 'Fare'):
            # 建立透视表。index表示关心的变量(也就是特征)，values是最终关注的结果。aggfunc是聚合函数，针对values建立的各种聚合函数
            print pd.pivot_table(dataSet, index=i, values='Survived', aggfunc=[np.sum, len, np.mean])
    print pd.pivot_table(dataSet, index='Pclass', values='Survived', aggfunc=[np.sum, len, np.mean])

    data = readData('../Titanic/data/train.csv')
    test_data = readData('../Titanic/data/test.csv')
    dataPredeal(data)
    dataPredeal(test_data)
    x, features = getX(data)
    print features
    label = getLabel(data)
    input_x, features = getX(test_data)
    x_result = calResult(x, label, input_x)
    # 由于test没有label，所以暂时使用训练数据看AUC
    performance(calResult(x, label, x), label)
    res = [[test_data["PassengerId"][i], x_result[i]] for i in range(len(x_result))]
    res.insert(0, ["PassengerId", "Survived"])
    writeData('result.csv', res)
