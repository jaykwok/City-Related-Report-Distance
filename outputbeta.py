from pandas import Series,DataFrame
from sklearn import linear_model
import pandas as pd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

#设置基本参数
K=1

#设置基本变量
I=[]
D=[]
Cii=[]
Cjj=[]
R2=[]
X=[]#回归模型的x坐标


#输入count文件
print('select count file in csv format\n')
root = tk.Tk()
root.withdraw()
file_path_count = filedialog.askopenfilename()
data_count=pd.read_csv(file_path_count)

#输入位置文件
print('select position file in csv format\n')
root = tk.Tk()
root.withdraw()
file_path_pos = filedialog.askopenfilename()
data_pos=pd.read_csv(file_path_pos)

#输入Cij和城市顺序文件
print('select Cij file in csv format\n')
root = tk.Tk()
root.withdraw()
file_path_comp = filedialog.askopenfilename()
data_comp=pd.read_csv(file_path_comp)


beta=np.linspace(0,1,101)#设置beta待选值

Cij = np.array(data_comp.loc[:, 'Cij'])

Rmax=0
Best_Beta=0

for i in beta:
    j=0#设置迭代器并且在一种beta算完之后重置迭代器j
    for index,row in data_comp.iterrows():
        city_i=row['city_a']
        city_j = row['city_b']
        vector1=np.array([np.float64(data_pos.loc[data_pos['name_eng']==city_i].x),np.float64(data_pos.loc[data_pos['name_eng']==city_i].y)])#记录city_a的坐标
        vector2=np.array([np.float64(data_pos.loc[data_pos['name_eng']==city_j].x),np.float64(data_pos.loc[data_pos['name_eng']==city_j].y)])#记录city_b的坐标
        D.append(np.linalg.norm(vector1-vector2))#将city_i到j的距离填充进列表
        Pi=np.int(data_count.loc[data_count['city']==city_i].newsCount)#获取Cii
        Pj = np.int(data_count.loc[data_count['city'] == city_j].newsCount)#获取Cjj
        Iij=K*Pi*Pj/np.power(D[j],i)
        X.append([Iij])
        I.append(Iij)#计算Iij并填充进列表
        Cii.append(Pi)#填充Cii
        Cjj.append(Pj)#填充Cjj
        j=j+1
    temp={"distance":D,"cii":Cii,"cjj":Cjj,"Iij":I}#拼接几个变量列表
    mix=DataFrame(temp)#转换为数据框架结构
    line=linear_model.LinearRegression()
    line.fit(X,Cij)
    Rtemp=line.score(X,Cij)
    R2.append(Rtemp)
    if (Rtemp>Rmax):
        Rmax=Rtemp
        Best_Beta=i
        prediction = line.predict(X)
    result=pd.concat([data_comp,mix],axis=1)#按照列来拼接数据框架
    name='result_beta%.2f.csv'%i#批量命名输出的CSV文件的文件名
    result.to_csv(name, index=False)#输出CSV文件
    D.clear()#清空距离列表
    Cii.clear()#清空Cii列表
    Cjj.clear()#清空Cjj列表
    I.clear()#清空Iij列表
    X.clear()#清空回归模型的x轴坐标
    mix.drop(mix.index, inplace=True)#清空转换的数据框架
    result.drop(result.index, inplace=True)#清空拼接的最终版数据框架

print('Calculated!\n')

readfilename='result_beta%.2f.csv'%Best_Beta
rdfile=pd.read_csv(readfilename)#读取最佳R2的值的CSV文件

print('the best Beta = %f\nthe best R = %f'%(Best_Beta,Rmax))

plt.figure(1)
plt.plot(beta,R2)
plt.xlabel('β')
plt.ylabel('R Squared')

plt.figure(2)
plt.scatter(rdfile.loc[:,'Iij'],rdfile.loc[:,'Cij'],label='Cij and Iij')
plt.plot(rdfile.loc[:,'Iij'],prediction,color="black",linewidth=.5,label='Regression Graph')
plt.legend(loc='upper left')
x_max=np.max(np.array(rdfile.loc[:,'Iij']))
plt.xlabel('Iij')
plt.ylabel('Cij')
plt.text(x_max/2,0,'the best Beta = %.2f\nthe best R^2 = %f'%(Best_Beta,Rmax),horizontalalignment='left')

plt.show()
