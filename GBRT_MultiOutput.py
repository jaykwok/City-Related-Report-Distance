from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

gbrt=ensemble.GradientBoostingRegressor(n_estimators=100)
reg=MultiOutputRegressor(gbrt)

raw=pd.read_csv('input.csv')

d=np.array(raw.loc[:,'distance'])
Pi=np.array(raw.loc[:,'cii'])
Pj=np.array(raw.loc[:,'cjj'])
Cij=np.array(raw.loc[:,'Cij'])

X=np.concatenate([d,Cij]).reshape(-1,2,order='F')
Y=np.concatenate([Pi,Pj]).reshape(-1,2,order='F')


reg.fit(X,Y)

predtion=reg.predict(X)

data=pd.read_csv('test.csv')

d_test=np.array(data.loc[:,'distance'])
Pi_test=np.array(data.loc[:,'cii'])
Pj_test=np.array(data.loc[:,'cjj'])
Cij_test=np.array(data.loc[:,'Cij'])

test=np.concatenate([d_test,Cij_test]).reshape(-1,2,order='F')
Y_test=np.concatenate([Pi_test,Pj_test]).reshape(-1,2,order='F')

prediction_test=reg.predict(test)
np.set_printoptions(precision=1, suppress=True)
prediction_test=np.array(prediction_test)
print('prediction is \n',prediction_test)
print('raw data is \n',Y_test)

print('Regression Score is ',reg.score(X,Y))