# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:02:56 2022

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('bitcoin.csv',parse_dates=['Date','Timestamp'],index_col=('Number'))

df['Date']=pd.to_numeric(df['Date'])/10000000000000
df['Timestamp']=pd.to_numeric(df['Timestamp'])

df.info()
df.isnull()
df.nunique()


X=df.iloc[:,[0,1,2,3,4,5,6,8]].values

y=df.iloc[:,7].values



from sklearn.impute import SimpleImputer
simpleImputer=SimpleImputer(missing_values=np.nan,strategy='mean')
simpleImputer.fit(X)
X=simpleImputer.transform(X)

#missing values
simpleImputer=SimpleImputer(missing_values=0,strategy='mean')
simpleImputer.fit(X)
X=simpleImputer.transform(X)


#label encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
y=labelEncoder.fit_transform(y)

'''
from sklearn.preprocessing import OneHotEncoder
columntransformer=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder= 'passthrough')
y=np.array(columntransformer.fit_transform(y))
'''

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=(0))

#scaler values
from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
X_train=standardScaler.fit_transform(X_train)
X_test=standardScaler.fit_transform(X_test)


from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train,y_train)
y_pred=linearRegression.predict(X_test)


'''
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,linearRegression.predict(X_test),color='blue')
plt.title('bitcoins')
plt.Xlabel('test')
plt.ylabel('Expectations')
plt.show()
'''


import sklearn.metrics as sm

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))



X=np.append(np.ones((len(X),1)).astype(int),values=X,axis=1)




import statsmodels.api as sm
'''
X_opt=X[:,[0,1,2,3,4,5,6,7]]
regressor_new=sm.OLS(endog=y,exog=X_opt).fit()
regressor_new.summary()

X_opt=X[:,[0,1,2,3,4,5,6]]
regressor_new=sm.OLS(endog=y,exog=X_opt).fit()
regressor_new.summary()
'''



def reg_ols(X,y):
    columns=list(range(X.shape[1]))
    
    for i in range(X.shape[1]):
        X_opt=np.array(X[:,columns],dtype=float) 
        regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
        pvalues = list(regressor_ols.pvalues)
        d=max(pvalues)
        if (d>0.05):
            for k in range(len(pvalues)):
                if(pvalues[k] == d):
                    del(columns[k])  
    
    return(X_opt,regressor_ols)

X_opt,regressor_ols=reg_ols(X, y)
regressor_ols.summary()



from sklearn.model_selection import train_test_split
X_train_opt,X_test_opt,y_train_opt,y_test_opt=train_test_split(X_opt,y,test_size=0.2,random_state=(0))



from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train_opt,y_train_opt)
y_pred1=linearRegression.predict(X_test_opt)

print('#'*50)

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test_opt, y_pred1), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test_opt, y_pred1), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test_opt, y_pred1), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test_opt, y_pred1), 2)) 
print("R2 score =", round(sm.r2_score(y_test_opt, y_pred1), 2))



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)

'''
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, pol_reg.predict(poly_reg.fit_transform(X_train)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''


import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test_opt, y_pred1), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test_opt, y_pred1), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test_opt, y_pred1), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test_opt, y_pred1), 2)) 
print("R2 score =", round(sm.r2_score(y_test_opt, y_pred1), 2))










