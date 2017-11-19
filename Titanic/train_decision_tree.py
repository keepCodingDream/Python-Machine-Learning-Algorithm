# coding:UTF-8

"""
  @Date 2017-11-18
  @author tracy
"""
import pydot
from sklearn import tree
from sklearn.externals import joblib
from sklearn.externals.six import StringIO

from train_svm import readData, dataPredeal, getLabel, getX, performance, writeData


def decisionTreeTrain(data, label):
    clf = tree.DecisionTreeClassifier()
    return clf.fit(data, label)


if __name__ == "__main__":
    data = readData('train.csv')
    test_data = readData('test.csv')
    dataPredeal(data)
    dataPredeal(test_data)
    x, features = getX(data)
    print features
    label = getLabel(data)
    input_x, features = getX(test_data)
    tree_result = decisionTreeTrain(x, label)
    dot_data = StringIO()
    tree.export_graphviz(tree_result, out_file=dot_data, feature_names=list(features), filled=True,
                         rounded=True, special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_dot('iris_simple.dot')
    graph[0].write_png('iris_simple.png')
    # 保存决策树模型
    joblib.dump(tree_result, "decision_tree_model.m")
    # 取出保存好的模型(训练)
    clf = joblib.load("decision_tree_model.m")
    decision_result_test = clf.predict(input_x)
    decision_result_train = clf.predict(x)
    # 由于test没有label，所以暂时使用训练数据看AUC
    performance(decision_result_train, label)
    res = [[test_data["PassengerId"][i], decision_result_test[i]] for i in range(len(decision_result_test))]
    res.insert(0, ["PassengerId", "Survived"])
    writeData('result_decision_tree.csv', res)
