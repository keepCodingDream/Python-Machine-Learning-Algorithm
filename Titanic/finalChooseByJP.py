# coding:UTF-8

"""
  @Date 2017-11-29
  @author tracy
"""

import csv
import math

import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


def extractTitleByName(combined_train):
    """
    从训练集、测试集中的name字段提取Title
    :param combined_train: 合并后的训练集
    :return: Title向量
    """
    combined_train['Title'] = combined_train['Name'].str.extract('.+,(.+)').str.extract('^(.+?)\.').str.strip()
    title_dict = {}
    title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_dict.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_dict.update(dict.fromkeys(['Master'], 'Master'))
    combined_train['Title'] = combined_train['Title'].map(title_dict)
    return combined_train


def trainMissingAges(missing_age_train):
    """
    根据已有参数训练预测Age模型,训练完以后，模型保存到/model目录下
    :param missing_age_train:
    :return:
    """
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    # model 1 GradientBoostingRegressor
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf = clf.fit(missing_age_X_train, missing_age_Y_train)
    joblib.dump(clf, "../Titanic/model/Age_GradientBoostingRegressor.m")
    mse = mean_squared_error(missing_age_Y_train, clf.predict(missing_age_X_train))
    print "GradientBoostingRegressor error:", mse
    scores = cross_val_score(clf, missing_age_X_train, missing_age_Y_train, cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    # model 2 LinearRegression
    lrf_reg = LinearRegression()
    lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
    lrf_reg_grid = GridSearchCV(lrf_reg, lrf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                scoring='neg_mean_squared_error')
    lrf_reg_grid = lrf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    joblib.dump(lrf_reg_grid, "../Titanic/model/Age_GridSearchCV.m")
    print 'Age feature Best LR Score:' + str(lrf_reg_grid.best_score_)
    print 'LR Train Error for "Age" Feature Regressor' + \
          str(lrf_reg_grid.score(missing_age_X_train, missing_age_Y_train))
    scores = cross_val_score(lrf_reg_grid, missing_age_X_train, missing_age_Y_train, cv=5)
    print "Regressor Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    # too bad not use it


def normalizationParams(combined_train):
    """
    将数据做归一化处理
    :param combined_train: 需要处理的数据
    :return: 归一化以后的数据
    """
    combined_train['Embarked'] = combined_train['Embarked'].fillna('S')
    combined_train['Embarked'] = combined_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    combined_train['Fare'] = combined_train[['Fare']].fillna(combined_train.groupby('Pclass').transform('mean'))
    combined_train.loc[combined_train['Fare'] <= 7.91, 'Fare'] = 0
    combined_train.loc[(combined_train['Fare'] > 7.91) & (combined_train['Fare'] <= 14.454), 'Fare'] = 1
    combined_train.loc[(combined_train['Fare'] > 14.454) & (combined_train['Fare'] <= 31), 'Fare'] = 2
    combined_train.loc[combined_train['Fare'] > 31, 'Fare'] = 3
    combined_train['Fare'] = combined_train['Fare'].astype(int)
    combined_train['Sex'] = combined_train['Sex'].fillna('male')
    combined_train['Sex'] = combined_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
    combined_train['Title'] = combined_train['Title'].map(
        {'Mrs': 0, 'Miss': 1, 'Mr': 2, 'Master': 3, 'Royalty': 4, "Officer": 5}).astype(int)
    return combined_train


def fillTheMissingAge(data):
    """
    fill the age is null
    :param data:
    :return:
    """
    clf = joblib.load('../Titanic/model/Age_GridSearchCV.m')
    test_data = pd.DataFrame(data[['Parch', 'Sex', 'SibSp', 'Title', 'Fare', 'Pclass', 'Embarked']])
    age_predicted = clf.predict(test_data)
    age = []
    for i in data.index:
        if math.isnan(data['Age'][i]):
            age.append(age_predicted[i])
        else:
            age.append(data['Age'][i])
    test_data['Age'] = age
    return test_data


def train_random_forest(data, label):
    rf_est = ensemble.RandomForestClassifier(random_state=42)
    rf_param_grid = {'n_estimators': [400], 'min_samples_split': [2, 3], 'max_depth': [40]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid = rf_grid.fit(data, label)
    # 将feature按Importance排序
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(data), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(20)['feature']
    print 'Sample 25 Features from RF Classifier:'
    print str(features_top_n_rf[:])
    scores = cross_val_score(rf_grid, data, label, cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    joblib.dump(rf_grid, "../Titanic/model/Survived_GradientBoostingRegressor.m")


def writeData(fileName, data):
    csvFile = open(fileName, 'w')
    writer = csv.writer(csvFile)
    n = len(data)
    for i in range(n):
        writer.writerow(data[i])
    csvFile.close()


if __name__ == "__main__":
    dataSet_train = pd.read_csv('../Titanic/data/train.csv')
    dataSet_test = pd.read_csv('../Titanic/data/test.csv')
    dataSet_train = dataSet_train.drop(0)
    dataSet_test = dataSet_test.drop(0)
    label = dataSet_train['Survived']
    # depend on the analysis we choose the features below
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # Step1 fill the null values
    dataSet_train['Embarked'] = dataSet_train['Embarked'].fillna('S')
    dataSet_test['Embarked'] = dataSet_test['Embarked'].fillna('S')
    # Step2 fill the Age by train
    # 2.1 extract the title by name
    combined_train = extractTitleByName(dataSet_train.append(dataSet_test))
    # 2.2 normalization the params
    combined_train = normalizationParams(combined_train)
    missing_age_df = pd.DataFrame(
        combined_train[['Age', 'Parch', 'Sex', 'SibSp', 'Title', 'Fare', 'Pclass', 'Embarked']])
    # make the ages existed as the train example.And make the missing as the test
    missing_age_train = missing_age_df[pd.notnull(missing_age_df['Age'])]
    missing_age_test = missing_age_df[pd.isnull(missing_age_df['Age'])]
    # trainMissingAges(missing_age_train)
    # 3.after train the age model,we will fill the None value
    # 3.1 at the first normalizationParams
    dataSet_train = extractTitleByName(dataSet_train)
    dataSet_train = normalizationParams(dataSet_train)
    dataSet_train = fillTheMissingAge(dataSet_train)
    # 3.2 normalize age
    dataSet_train.loc[dataSet_train['Age'] <= 16, 'Age'] = 0
    dataSet_train.loc[(dataSet_train['Age'] > 16) & (dataSet_train['Age'] <= 32), 'Age'] = 1
    dataSet_train.loc[(dataSet_train['Age'] > 32) & (dataSet_train['Age'] <= 48), 'Age'] = 2
    dataSet_train.loc[(dataSet_train['Age'] > 48) & (dataSet_train['Age'] <= 64), 'Age'] = 3
    dataSet_train.loc[dataSet_train['Age'] > 64, 'Age'] = 4
    train_random_forest(dataSet_train, label)
    # 4 final get result
    dataSet_test = pd.read_csv('../Titanic/data/test.csv')
    passengerId = dataSet_test['PassengerId']
    dataSet_test = extractTitleByName(dataSet_test)
    dataSet_test = normalizationParams(dataSet_test)
    dataSet_test = fillTheMissingAge(dataSet_test)
    final_model = joblib.load("../Titanic/model/Survived_GradientBoostingRegressor.m")
    result = final_model.predict(dataSet_test)
    dataSet_test['PassengerId'] = passengerId
    res = [[dataSet_test["PassengerId"][i], result[i]] for i in range(len(result))]
    res.insert(0, ["PassengerId", "Survived"])
    writeData('result.csv', res)
