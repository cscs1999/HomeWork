import numpy as np
import math
import matplotlib.pyplot as plt
import time

#参数列表
totalTimes=1500000
lr = 0.00015
N1 = 50
N2 = 10
times = []
trainError = []
testError = []

#读取数据集
def read(f, n, x, y):
    for i in range(n):
        line = f.readline()
        x[0][i], x[1][i], y[0][i] = list(map(float, line.split()))
        x[2][i] = 1

#打印并保存图像
def printPic(x, y1, y2, name):
    plt.figure()
    plt.plot(x, y1, c='b', linewidth=1, label="trainError")
    plt.plot(x, y2, c='r', linewidth=1, label="testError")
    plt.xlabel("iterationTimes")  # X轴标签
    plt.ylabel("Error")   # Y轴标签
    plt.legend()    # 显示图例
    plt.savefig(name)
    plt.show()
    
trainData = open("DataSet/dataForTrainingLinear.txt")
testData = open("DataSet/dataForTestingLinear.txt")

trainx = np.zeros((3, N1), dtype=np.float)
trainy = np.zeros((1, N1), dtype=np.float)
w = np.zeros((1, 3), dtype=np.float)
testx = np.zeros((3, N2), dtype=np.float)
testy = np.zeros((1, N2), dtype=np.float)
read(trainData, N1, trainx, trainy)
read(testData, N2, testx, testy)

start = time.time()
for i in range(totalTimes):
    fx = np.dot(w, trainx) #矩阵乘法
    w = w - lr * np.dot((fx - trainy), trainx.T) / N1 #梯度下降法的迭代公式
    if i % 100000 == 0 and i > 0:
        trainError.append(np.sum((np.dot(w, trainx) - trainy) ** 2) / N1) #计算训练误差
        testError.append(np.sum((np.dot(w, testx) - testy) ** 2) / N2) #计算测试误差
        times.append(i)

printPic(times, trainError, testError, "iterationTimes&Error_a.png")
end = time.time()
print(end-start)
