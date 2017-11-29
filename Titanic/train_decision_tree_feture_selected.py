# coding:UTF-8

"""
  @Date 2017-11-18
  @author tracy
"""
import numpy as np
import pandas as pd
import pydotplus
from sklearn import feature_selection
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pylab as pl

from train_svm import writeData

if __name__ == "__main__":
    titanic = pd.read_csv('train.csv')
    titanic_test = pd.read_csv('test.csv')

    # 先取出'pclass', 'age', 'sex'三个特征
    X = titanic.drop(['Name', 'Survived'], axis=1)
    y = titanic['Survived']

    t_X = titanic_test.drop(['Name'], axis=1)

    # 首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略。
    X['Age'].fillna(X['Age'].mean(), inplace=True)
    t_X['Age'].fillna(X['Age'].mean(), inplace=True)
    X.fillna('UNKNOWN', inplace=True)
    t_X.fillna('UNKNOWN', inplace=True)

    # 由于测试数据没有标签，所以将训练数据随机分成成1:3的比例
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    vec = DictVectorizer(sparse=False)
    # 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    # 同样需要对测试数据的特征进行转换。
    X_test = vec.transform(X_test.to_dict(orient='record'))

    percentiles = range(1, 100, 5)
    results = []

    clf = DecisionTreeClassifier(max_depth=20, min_samples_leaf=3)

    for i in percentiles:
        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
        X_train_fs = fs.fit_transform(X_train, y_train)
        scores = cross_val_score(clf, X_train_fs, y_train, cv=5)
        results = np.append(results, scores.mean())
    print results

    opt = np.where(results == results.max())[0]
    print opt[0]

    pl.plot(percentiles, results)
    pl.xlabel('percentiles of features')
    pl.ylabel('accuracy')
    pl.show()

    # 使用分割到的训练数据进行模型学习。
    tree_result = clf.fit(X_train, y_train)
    # 用训练好的决策树模型对测试特征数据进行预测。
    y_predict = clf.predict(X_test)

    # 输出预测准确性。
    print(clf.score(X_test, y_test))
    # 输出更加详细的分类性能。
    print(classification_report(y_predict, y_test, target_names=['Died', 'Survived']))
    # 保存决策树模型
    joblib.dump(tree_result, "decision_tree_model.m")
    # 取出保存好的模型(训练)
    clf = joblib.load("decision_tree_model.m")
    t_X = vec.transform(t_X.to_dict(orient='record'))
    decision_result_test = clf.predict(t_X)
    dot_data = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("decision_tree_model.pdf")
    res = [[titanic_test["PassengerId"][i], decision_result_test[i]] for i in range(len(decision_result_test))]
    res.insert(0, ["PassengerId", "Survived"])
    writeData('result_decision_tree.csv', res)
