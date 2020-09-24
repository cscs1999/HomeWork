import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.stdout = open('Exercise3.log', mode = 'w',encoding='utf-8')

def f(x,y):
  return (y**2*np.exp(-y**2)+x**4*np.exp(-x**2))/(x*np.exp(-x**2))
a1 = 2;
b1 = 4;
a2 = -1;
b2 = 1;
# use N draws
Ns= [10,20,30,40,50,60,70,80,100,200,500]
Imc_lists = [[] for i in range(100)]  # 创建的是多行100列的二维列表
for i in range(100):
    for N in Ns:
        X = np.random.uniform(low=a1, high=b1, size=N) # N values uniformly drawn from a to b
        Y = np.random.uniform(low=a2, high=b2, size=N) # N values uniformly drawn from a to b
        Z =f(X,Y)  # CALCULATE THE f(x)
        # 蒙特卡洛法计算定积分：体积=长度*宽度*平均高度
        Imc= (b1-a1)*(b2-a2)*np.sum(Z)/N;
        Imc_lists[i].append(Imc)
df = pd.DataFrame(Imc_lists)
df.columns = ['10','20','30','40','50','60','70','80','100','200','500']
df.index = range(1,len(df) + 1) # 将index改成从1开始
# 自定义数字转换为序数词函数
def add_ordinal(x):
    if x%10==1 and x!=11:
        return str(x)+'st'
    elif x%10==2 and x!=12:
        return str(x)+'nd'
    elif x%10==3 and x!=13:
        return str(x)+'rd'
    else:
        return str(x)+'th'
for para in['10','20','30','40','50','60','70','80','100','200','500']:
    df.loc[101,para]=df.loc[:100,para].mean()
    df.loc[102,para]=df.loc[:100,para].var()
df=df.rename(index=add_ordinal)
df=df.rename(index={'101st':'mean', '102nd':'variance'})

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('display.width', 5000)
# 保留小数点后1位
df=df.apply(lambda x:round(x,1))
print(df)
