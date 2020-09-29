'''
ggplot的优点是：
第一，有明确的起始（以ggplot函数开始）与终止（一句语句一幅图）；
其二，图层之间的叠加是靠“+”号实现的，越后面其图层越高；
其三，绘图更加美观。R基础包的绘图函数没有一个停止绘图的标志，这使得有时候再处理会产生一些困惑。

缺点：ggplot2如果要调整参数，则意味着要重新作图。

特点：
matplotlib：支持交互式和非交互式绘图。

可将图像保存成PNG 、PS等多种图像格式。

支持曲线（折线）图、条形图、柱状图、饼图。

图形可配置。

跨平台，支持Linux, Windows，Mac OS X与Solaris。

Matplotlib的绘图函数基本上都与MATLAB的绘图函数名字差不多，迁移学习的成本比较低。

支持LaTeX的公式插入。

seaborn:内置数个经过优化的样式效果。
增加调色板工具，可以很方便地为数据搭配颜色。
单变量和双变量分布绘图更为简单，可用于对数据子集相互比较。
对独立变量和相关变量进行回归拟合和可视化更加便捷。
对数据矩阵进行可视化，并使用聚类算法进行分析。
基于时间序列的绘制和统计功能，更加灵活的不确定度估计。
基于网格绘制出更加复杂的图像集合。

ggplot:
ggplot2的核心理念是将绘图与数据分离，数据相关的绘图与数据无关的绘图分离
ggplot2是按图层作图
ggplot2保有命令式作图的调整函数，使其更具灵活性
ggplot2将常见的统计变换融入到了绘图中。
'''

from plotnine import *
import seaborn as sb
from pandas import Series,DataFrame
import pandas as pd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

kinds=['Red wine','Liquor','Rice wine','Beer']
surveys=[3,4,1,1,3,4,3,3,1,3,2,1,2,1,3,4,1,1,3,4,3,3,1,3,2,1,2,1,2,3,2,3,1,1,1,1,4,3,1,2,3,2,3,1,1,1,1,4,3,1]
print("调查人数为:",len(surveys))
result=[]
for i in surveys:
    result.append(kinds[i-1])


plt.subplot(1,2,1)
plt.hist(result)
plt.title('Drawed by matplotlib')

plt.subplot(1,2,2)
sb.countplot(result)
plt.title('Drawed by seaborn')


data={'kind':result}
df=DataFrame(data)
print(df.kind)
p=ggplot(df, aes(x=df.kind))+geom_histogram(stat="bin",binwidth=.5)+labs(title = 'Drawed by ggplot')

print(p)

plt.show()