# coding:UTF-8

"""
  @Date 2017-12-04
  @author tracy
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载数据
iris = load_iris()
# 基本数据
data = iris.data
# 标签
target = iris.target

# 标准化数据
data = StandardScaler().fit_transform(data)
data = MinMaxScaler().fit_transform(data)
data = Normalizer().fit_transform(data)

# 对定量特征二值化(这里对第一个特征二值化了)
data_copy = data[:, 0:1]
data_copy = Binarizer(threshold=0.3).fit_transform(data_copy)
data[:, 0:1] = data_copy



