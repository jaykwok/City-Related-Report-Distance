from sklearn import neural_network
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

reg=ensemble.GradientBoostingRegressor(n_estimators=100)

raw=pd.read_csv('input.csv')

d=np.array(raw.loc[:,'distance'])
Pi=np.array(raw.loc[:,'cii'])
Pj=np.array(raw.loc[:,'cjj'])

X=np.concatenate([d,Pi,Pj]).reshape(-1,3,order='F')
Y=np.array(raw.loc[:,'Cij'])

reg.fit(X,Y)

predtion=reg.predict(X)

data=pd.read_csv('test.csv')

d_test=np.array(data.loc[:,'distance'])
Pi_test=np.array(data.loc[:,'cii'])
Pj_test=np.array(data.loc[:,'cjj'])

test=np.concatenate([d_test,Pi_test,Pj_test]).reshape(-1,3,order='F')
Y_test=np.array(data.loc[:,'Cij'])

prediction_test=reg.predict(test)
prediction_test=np.array(prediction_test)
print('prediction is \n',prediction_test)
print('raw data is \n',Y_test)

mixed={'Test Data':Y_test,'Prediction Data':prediction_test}
comp=DataFrame(mixed)

comp.to_excel('compare.xlsx',index=False)

plt.plot(Y_test,label='Test Raw Data')
plt.plot(prediction_test,label='Predicton Data')
plt.legend(loc='upper right')
plt.title('Regression Score is %f'%reg.score(X,Y))
plt.show()

print('Regression Score is ',reg.score(X,Y))

