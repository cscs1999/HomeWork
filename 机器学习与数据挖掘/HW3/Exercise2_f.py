import numpy as np
import math
import matplotlib.pyplot as plt
import random

#参数列表
totalTimes=40
lr = 0.00015
N1 = 400 #训练集大小
N2 = 100 #测试集大小
N3 = 7 #属性个数
times = []
trainErrorRate = []
testErrorRate = []

#读取数据集
def read(f, n, x, y):
    for i in range(n):
        line = f.readline()
        a = list(map(float, line.split()))
        for j in range(6):
            x[j][i] = a[j]
        y[0][i] = a[6]
        x[6][i] = 1

#打印并保存图像
def printPic(x, y1, y2, name):
    plt.figure()
    plt.plot(x, y1, c='b', linewidth=1, label="trainError")
    plt.plot(x, y2, c='r', linewidth=1, label="testError")
    plt.xlabel("trainingSize") # X轴标签
    plt.ylabel("errorRate")   # Y轴标签
    plt.legend()    # 显示图例
    plt.savefig(name, dpi=300)
    plt.show()

trainData = open("DataSet/dataForTrainingLogistic.txt")
testData = open("DataSet/dataForTestingLogistic.txt")

trainx = np.zeros((N3, N1), dtype=np.float)
trainy = np.zeros((1, N1), dtype=np.float)
w = np.zeros((1, N3), dtype=np.float)
testx = np.zeros((N3, N2), dtype=np.float)
testy = np.zeros((1, N2), dtype=np.float)
read(trainData, N1, trainx, trainy)
read(testData, N2, testx, testy)

a = list(range(N1))
for k in range(totalTimes):
    tempSize = (k + 1) * 10
    b = random.sample(a, tempSize)
    x = np.zeros((N3, tempSize), dtype=np.float)
    y = np.zeros((1, tempSize), dtype=np.float)
    w = np.zeros((1, N3), dtype=np.float)
    
    for i in range(tempSize):
        for j in range(N3):
            x[j][i] = trainx[j][b[i]]
        y[0][i] = trainy[0][b[i]]
    
    for i in range(100):
        fx = np.dot(w, x)
        fx = np.exp(fx)
        fx = fx / (1 + fx)
        w = w - lr * np.dot((fx - y), x.T)
        fx = np.dot(w, x)

    ans = 0
    fx = np.dot(w, x)
    fx = np.exp(fx)
    fx = fx / (1 + fx)
    for i in range(tempSize):
        res = 0
        if fx[0][i] > 0.5:
            res = 1
        if res != y[0][i]:
            ans += 1
        #print(fx[0][i])
    trainErrorRate.append(float(ans)/float(tempSize))

    ans = 0
    fx = np.dot(w, testx)
    fx = np.exp(fx)
    fx = fx / (1 + fx)
    for i in range(100):
        res = 0
        if fx[0][i] > 0.5:
            res = 1
        if res != testy[0][i]:
            ans += 1
    testErrorRate.append(float(ans)/float(tempSize))
    times.append(tempSize)

printPic(times, trainErrorRate, testErrorRate, "trainingSize&ErrorRate_f.png")
