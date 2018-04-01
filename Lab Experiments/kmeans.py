import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from kmodes.kmodes import KModes

iris=load_iris()
X=iris.data
y=iris.target
names=iris.target_names
df=pd.DataFrame(X,columns=['Sepal-Length','Sepal-Width','Petal-Length','Petal-Width'])
df.to_csv('iris.csv')

clf=KMeans(n_clusters=3,init='k-means++',n_init=10,random_state=10,max_iter=300).fit(X)
km=KModes(n_clusters=3,init='Huang',n_init=5,verbose=1)
kmpre=km.fit_predict(X)
pre=clf.predict(X)

def plot_testdata(X,pre,clf):
    plt.scatter(X[pre == 0, 0], X[pre == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(X[pre == 1, 0], X[pre == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
    plt.scatter(X[pre == 2, 0], X[pre == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
    plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
    plt.legend()
    plt.show()

print ("Accuracy for the current KMeans model is: ",-clf.score(X))
plot_testdata(X,pre,clf)
print (kmpre)