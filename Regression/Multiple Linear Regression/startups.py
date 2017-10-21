import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

labelX=LabelEncoder()
X[:,3]=labelX.fit_transform(X[:,3])
encoder=OneHotEncoder(categorical_features=[3])
X=encoder.fit_transform(X).toarray()

#Dummy Variable Trap
X=X[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#Optimal variable which contains statistically significant variable
X_opts=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opts).fit()
regressor_OLS.summary()