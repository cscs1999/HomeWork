import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
sys.stdout = open('Exercise1.log', mode = 'w',encoding='utf-8')
# 投点次数
ns=[20,50,100,200,300,500,1000,5000]
# 圆的半径、圆心
r = 1
a,b = (0,0)
# 正方形区域
x_min, x_max = a, a+r
y_min, y_max = b, b+r
# 记录pi值的list
pi_lists = [[] for i in range(20)]  # 创建的是多行20列的二维列表
for i in range(20):
    # print("第%d次实验结果:"%(i+1))
    # 在正方形区域内随机投点
    fig,axes = plt.subplots(2,5,figsize=(15,5))
    plt.subplots_adjust(left=1, bottom=1, right=2, top=2, wspace=0.5, hspace=0.5)
    # 计数器
    counter=0
    for n in ns:
        x = np.random.uniform(x_min, x_max, n) #均匀分布
        y = np.random.uniform(y_min, y_max, n)
        # 计算点到圆心的距离
        d = np.sqrt((x-a)**2 + (y-b)**2)
        # 统计落在圆内点的数目
        res = sum(np.where(d < r, 1, 0))
        x1= x[np.where(d<r)]
        x2= x[np.where(d>r)]
        y1= y[np.where(d<r)]
        y2= y[np.where(d>r)]
        # 计算pi的近似值（Monte Carlo:用统计值去近似真实值）
        pi = 4 * res / n
        # print('pi: ',pi)
        pi_lists[i].append(pi)
        # 可视化
        axes_temp = axes[int(counter/4),int(counter%4)]
        axes_temp.plot(x, y,'ro',markersize = 2)
        axes_temp.plot(x2, y2,'go',markersize = 2)
        circle = Circle(xy=(a,b), radius=r, alpha=0.5)
        axes_temp.add_patch(circle)
        plt.axis('equal') # 防止图像变形
        plt.axis([0.6, 1, 0, 1])
        counter = counter+1
    plt.show()
df=pd.DataFrame(pi_lists)
df.columns = ['20','50','100','200','300','500','1000','5000']
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
for para in['20','50','100','200','300','500','1000','5000']:
    df.loc[21,para]=df.loc[:20,para].mean()
    df.loc[22,para]=df.loc[:20,para].var()
df=df.rename(index=add_ordinal)
df=df.rename(index={'21st':'mean', '22nd':'variance'})
exact_lists=[]
for i in range(21):
    exact_lists.append(3.14159)
exact_lists.append(0)
df['exact'] = exact_lists
pd.set_option('display.width', 800)
# 保留小数点后5位
df=df.apply(lambda x:round(x,5))
print(df)
