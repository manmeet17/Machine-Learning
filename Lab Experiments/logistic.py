import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve,precision_recall_fscore_support,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer


data=load_breast_cancer()
feat=data.feature_names
X=pd.DataFrame(data.data,columns=feat)
y=pd.DataFrame(data.target)
df=pd.concat([X,y],axis=1)
df=df.rename(columns={0:'Class'})
df.to_csv('breast_cancer.csv')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=pd.DataFrame(X_train,columns=feat)
X_test=pd.DataFrame(X_test,columns=feat)

clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

assert len(y_pred)==len(y_test)

print ("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print ("\nAccuracy: ",accuracy_score(y_test,y_pred))

precision, recall, fscore, support=precision_recall_fscore_support(y_test,y_pred,average='binary')

print ("\nPrecision: ",precision)
print("\nRecall: ",recall)
print ("\nF1 Score: ",fscore)


def plot_pr(y_test,y_pred):
    average_precision = average_precision_score(y_test, y_pred)
    print('\nAverage precision-recall score: {0:0.2f}'.format(average_precision))
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure(figsize=(20,10))
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

def plot_data(X,y):
    sns.distplot(y,bins=2,kde=False,rug=True)
    plt.title('Diagnosis (M=1 , B=0)')
    plt.show()

plot_data(X,y)
plot_pr(y_test,y_pred)


