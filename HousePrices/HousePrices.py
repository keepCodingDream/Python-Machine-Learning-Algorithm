# coding:UTF-8

"""
  @Date 2017-12-04
  @author tracy
"""
import csv

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def log_format(dataSet_train):
    dataSet_train['GrLivArea'] = np.log(dataSet_train['GrLivArea'])
    # TotalBsmtSF二值化，平滑过渡没有地下室的影响
    dataSet_train['HasBsmt'] = pd.Series(len(dataSet_train['TotalBsmtSF']), index=dataSet_train.index)
    dataSet_train['HasBsmt'] = 0
    dataSet_train.loc[dataSet_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    dataSet_train.loc[dataSet_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(dataSet_train['TotalBsmtSF'])
    return dataSet_train


def writeData(fileName, data):
    csvFile = open(fileName, 'w')
    writer = csv.writer(csvFile)
    n = len(data)
    for i in range(n):
        writer.writerow(data[i])
    csvFile.close()


if __name__ == "__main__":
    dataSet_train = pd.read_csv('../HousePrices/data/train.csv')
    dataSet_test = pd.read_csv('../HousePrices/data/test.csv')
    # 根据HousePrices.ipynb的分析，选择OverallQual\GrLivArea\GarageArea\TotalBsmtSF
    price = dataSet_train['SalePrice']
    dataSet_train = dataSet_train[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']]
    price = np.array(np.log1p(price)).tolist()
    # 开始对数变换
    dataSet_train = log_format(dataSet_train)
    dataSet_train = pd.get_dummies(dataSet_train)
    linearRegression = LinearRegression()
    model_linear = linearRegression.fit(dataSet_train, price)
    joblib.dump(model_linear, "../HousePrices/model/linear.m")
    scores = cross_val_score(model_linear, dataSet_train, price, cv=5)
    print "linearRegression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    mse = mean_squared_error(price, model_linear.predict(dataSet_train))
    print "linearRegression mean_squared_error:", mse

    # random forest
    # max_features = [.1, .3, .5, .7, .9, .99]
    # test_scores = []
    # for max_feat in max_features:
    #     clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    #     test_score = np.sqrt(-cross_val_score(clf, dataSet_train, price, cv=5, scoring='neg_mean_squared_error'))
    #     test_scores.append(np.mean(test_score))

    # print test_scores
    # test_scores result is [0.17069660489633223, 0.17164344976207629, 0.17160985933866174, 0.17268295648085472, 0.17452645522665336, 0.17406118029269227]
    # so we choose max_features=.99
    model_for_reg = RandomForestRegressor(n_estimators=200, max_features=.99)
    model_for_reg = model_for_reg.fit(dataSet_train, price)
    joblib.dump(model_for_reg, "../HousePrices/model/model_for_reg.m")
    scores = cross_val_score(model_for_reg, dataSet_train, price, cv=5)
    print "RandomForestRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    mse = mean_squared_error(price, model_for_reg.predict(dataSet_train))
    print "RandomForestRegressor mean_squared_error:", mse

    # start to get result
    id = dataSet_test['Id']
    dataSet_test = dataSet_test[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']]
    print dataSet_test.index[np.where(np.isnan(dataSet_test))[0]]
    print dataSet_test.columns[np.where(np.isnan(dataSet_test))[1]]
    dataSet_test['GarageArea'] = dataSet_test['GarageArea'].fillna(dataSet_test['GarageArea'].mean())
    dataSet_test['TotalBsmtSF'] = dataSet_test['TotalBsmtSF'].fillna(int(dataSet_test['TotalBsmtSF'].mean()))
    dataSet_test = log_format(dataSet_test)
    dataSet_test = pd.get_dummies(dataSet_test)

    # result = model_linear.predict(dataSet_test)
    result = model_for_reg.predict(dataSet_test)

    dataSet_test['Id'] = id
    res = [[dataSet_test["Id"][i], np.expm1(result[i])] for i in range(len(result))]
    res.insert(0, ["Id", "SalePrice"])
    writeData('result.csv', res)
