import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.stdout = open('Exercise2.log', mode = 'w',encoding='utf-8')

def f(x):
  return x**3
def intf(x):
  return x**4/4
a = 0;
b = 1;
# use N draws
Ns= [5,10,20,30,40,50,60,70,80,100]
Imc_lists = [[] for i in range(100)]  # 创建的是多行100列的二维列表
exactval=intf(b)-intf(a)
for i in range(100):
    for N in Ns:
        X = np.random.uniform(low=a, high=b, size=N) # N values uniformly drawn from a to b
        Y =f(X)  # CALCULATE THE f(x)
        # 蒙特卡洛法计算定积分：面积=宽度*平均高度
        Imc= (b-a) * np.sum(Y)/ N;
        Imc_lists[i].append(Imc)
df = pd.DataFrame(Imc_lists)
df.columns = ['5','10','20','30','40','50','60','70','80','100']
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
for para in['5','10','20','30','40','50','60','70','80','100']:
    df.loc[101,para]=df.loc[:100,para].mean()
    df.loc[102,para]=df.loc[:100,para].var()
df=df.rename(index=add_ordinal)
df=df.rename(index={'101st':'mean', '102nd':'variance'})
exact_lists=[]
for i in range(101):
    exact_lists.append(0.25)
exact_lists.append(0)
df['exact'] = exact_lists
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('display.width', 1000)
print(df)
