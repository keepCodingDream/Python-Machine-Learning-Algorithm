# coding:UTF-8
'''
Date:20160901
@author: zhaozhiyong
'''
import numpy as np


class LogisticRegression:
    maxCycle = None
    step = None

    def __init__(self, max_cycle=1000, step=0.01):
        self.maxCycle = max_cycle
        self.step = step

    def load_data(self, file_name):
        """
        导入训练数据
        input:  file_name(string)训练数据的位置
        output: feature_data(mat)特征
                label_data(mat)标签
        """
        f = open(file_name)  # 打开文件
        feature_data = []
        label_data = []
        for line in f.readlines():
            feature_tmp = []
            lable_tmp = []
            lines = line.strip().split("\t")
            feature_tmp.append(1)  # 偏置项
            for i in xrange(len(lines) - 1):
                feature_tmp.append(float(lines[i]))
            lable_tmp.append(float(lines[-1]))

            feature_data.append(feature_tmp)
            label_data.append(lable_tmp)
        f.close()  # 关闭文件
        '''
           在这里所有特征会根据行构造成一个N*(M+1)的矩阵.(其中N为数据行数，M为特征个数，+1为偏至项)；

           所有的结果会构造成一个N*1的矩阵。(N为数据行数，1代表这一行的分类)
        '''
        return np.mat(feature_data), np.mat(label_data)

    def sig(self, x):
        """
        Sigmoid函数
        input:  x(mat):feature * w
        output: sigmoid(x)(mat):Sigmoid值

        """
        return 1.0 / (1 + np.exp(-x))

    def lr_train_bgd(self, feature, label, maxCycle, alpha):
        """
        利用梯度下降法训练LR模型
        input:  feature(mat)特征
                label(mat)标签
                maxCycle(int)最大迭代次数
                alpha(float)学习率
        output: w(mat):权重
        """
        '''
          np.shape就是输出矩阵是几行几列
        '''
        n = np.shape(feature)[1]  # 特征个数
        w = np.mat(np.ones((n, 1)))  # 初始化权重
        i = 0
        while i <= maxCycle:  # 在最大迭代次数的范围内
            i += 1  # 当前的迭代次数
            h = self.sig(feature * w)  # 计算Sigmoid值
            err = label - h
            if i % 100 == 0:
                print "\t---------iter=" + str(i) + \
                      " , train error rate= " + str(self.error_rate(h, label))
            w += alpha * feature.T * err  # 权重修正
        return w

    def error_rate(self, h, label):
        '''计算当前的损失函数值
        input:  h(mat):预测值
                label(mat):实际值
        output: err/m(float):错误率
        '''
        m = np.shape(h)[0]

        sum_err = 0.0
        for i in xrange(m):
            if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
                sum_err -= (label[i, 0] * np.log(h[i, 0]) + (1 - label[i, 0]) * np.log(1 - h[i, 0]))
            else:
                sum_err -= 0
        return sum_err / m

    def save_model(self, file_name, w):
        '''保存最终的模型
        input:  file_name(string):模型保存的文件名
                w(mat):LR模型的权重
        '''
        m = np.shape(w)[0]
        f_w = open(file_name, "w")
        w_array = []
        for i in xrange(m):
            w_array.append(str(w[i, 0]))
        f_w.write("\t".join(w_array))
        f_w.close()


if __name__ == "__main__":
    # 1、导入训练数据
    print "---------- 1.load data ------------"
    train = LogisticRegression()
    feature, label = train.load_data("data.txt")
    # 2、训练LR模型
    print "---------- 2.training ------------"
    w = train.lr_train_bgd(feature, label, 1000, 0.01)
    # 3、保存最终的模型
    print "---------- 3.save model ------------"
    train.save_model("weights_new", w)
