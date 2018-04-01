import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
import math
from collections import Counter
import pickle

points=[[1,2],[2,1],[4,4],[5,3],[3,2],[0,5],[4,2],[3,7],[3,0],[5,6],[3,3],[6,4],[7,2]]
labels=[0,1,1,1,0,1,1,0,1,0,1,1,0]
k=4
with open("test_data.pickle","rb") as f:
    test=pickle.load(f)

# for i in range(15):
#     test.append(np.random.randint(0,11,size=2))
# with open("test_data.pickle","wb") as f:
#     pickle.dump(test,f)
true_labels=[0,0,1,1,1,0,0,1,1,0,1,1,0,1,0]
x=[]
y=[]
dist=[]
pred=[]
for i,j in test:
    for p,q in enumerate(points):
        d=math.sqrt((q[0]-i)**2+(q[1]-j)**2)
        dist.append((d,p))
        x.append(q[0])
        y.append(q[1])
        dist.sort()
    lab=[]
    for neighbors in range(k):
        lab.append(labels[dist[neighbors][1]])
    c=Counter(lab)
    print ("The data point was classified as:",c.most_common()[0][0])
    pred.append(c.most_common()[0][0])
    lab=[]
    dist=[]


# Evaluation
print ("Accuracy: ",accuracy_score(true_labels,pred))
print ("F1 Score: ",f1_score(true_labels,pred))

color=['red' if i==1 else 'green' for i in labels]
# print (color)
cont=0
plt.figure(figsize=(15,10))
plt.subplot(211)
for i,j in points:
    if cont==0 or cont ==1:
        plt.scatter(i,j,c=color[cont],label="Class: "+str(labels[cont]))
    else:
        plt.scatter(i,j,c=color[cont])
    cont+=1
# plt.scatter(x,y,c=color,s=50,label=list(set(color)))
plt.xlabel('X Points')
plt.ylabel('Y Points')
plt.legend(loc='best')
plt.title('Plotting the datapoints')


plt.subplot(212)
color=['red' if i==1 else 'green' for i in pred]
cont=0
for i,j in test:
    if cont==0 or cont ==1:
        plt.scatter(i,j,c=color[cont],label="Class: "+str(cont))
    else:
        plt.scatter(i,j,c=color[cont])
    cont+=1
plt.xlabel('X Points')
plt.ylabel('Y Points')
plt.legend(loc='best')
plt.title("Plotting test data")
plt.show()







