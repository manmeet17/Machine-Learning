import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split

#Reading data
data=pd.read_csv('Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#Missing data
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

# Encoding categorical data
labelX=LabelEncoder()
x[:,0]=labelX.fit_transform(x[:,0])
labelY=LabelEncoder()
y=labelX.fit_transform(y)
encoder=OneHotEncoder(categorical_features=[0])
x=encoder.fit_transform(x).toarray()

# Splitting dataset into training and testing
#test_size is generally 0.2 0.25 or 0.3 and rarely 0.4
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
#We dont have to do it for y because we only have 1 and 0 and not a range of values


print (pd.DataFrame(x))
print (pd.DataFrame(y))
