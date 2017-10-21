# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 21:34:12 2017

@author: rijhwani
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

regressor=LinearRegression()
#Independent and dependent variable are parameters of fit
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("Salary vs Experience(Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("Salary vs Experience(Testing Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()