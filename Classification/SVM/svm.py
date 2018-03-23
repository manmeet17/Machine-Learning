import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from matplotlib.colors import ListedColormap

def linear_svc(X_train,y_train,X_test):
    classifier=SVC(kernel='linear',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    return y_pred,classifier

def rbf_svc(X_train,y_train,X_test):
    classifier=SVC(kernel='rbf',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    return y_pred,classifier

def poly_svc(X_train,y_train,X_test):
    classifier=SVC(C=1,kernel='poly',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    return y_pred,classifier
    
def plot(X_train,y_train,classifier,name):
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(name+'Classifier')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_pred,clf=linear_svc(X_train,y_train,X_test)
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix for Linear SVM: \n",cm)
print ("\n Precision: ",precision_score(y_test,y_pred))
print ("\n Accuracy: ",accuracy_score(y_test,y_pred))
print ("\n Recall: ",recall_score(y_test,y_pred))
print ("\n F1 Score: ",f1_score(y_test,y_pred))
plot(X_train,y_train,clf,"Linear SVM")

y_pred,clf=rbf_svc(X_train,y_train,X_test)
cm = confusion_matrix(y_test, y_pred)
print ("\nConfusion Matrix for RBF kernel : \n",cm)
print ("\n Precision: ",precision_score(y_test,y_pred))
print ("\n Accuracy: ",accuracy_score(y_test,y_pred))
print ("\n Recall: ",recall_score(y_test,y_pred))
print ("\n F1 Score: ",f1_score(y_test,y_pred))
plot(X_train,y_train,clf,"RBF kernel ")

y_pred,clf=poly_svc(X_train,y_train,X_test)
cm = confusion_matrix(y_test, y_pred)
print ("\nConfusion Matrix for Polynomial kernel SVM:\n ",cm)
print ("\n Precision: ",precision_score(y_test,y_pred))
print ("\n Accuracy: ",accuracy_score(y_test,y_pred))
print ("\n Recall: ",recall_score(y_test,y_pred))
print ("\n F1 Score: ",f1_score(y_test,y_pred))
plot(X_train,y_train,clf,"Polynomial kernel SVM")

