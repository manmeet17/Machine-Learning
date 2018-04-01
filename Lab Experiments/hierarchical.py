import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

X,y=make_blobs(n_samples=200, n_features=2,centers=2)
# print (X,y)

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

clf=AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="average")
clf.fit(X)
labels=clf.labels_

cnt=0
for i,j in zip(y,labels):
    if i==j:
        cnt+=1
print("AGNES Clustering")
print ("Accuracy: ",str((cnt/len(y))*100)+'%')

print ("DIANA Clustering")


