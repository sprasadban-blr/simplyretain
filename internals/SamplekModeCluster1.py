'''
https://analyticsdefined.com/using-k-modes-clustering-categorical-data/
https://archive.ics.uci.edu/ml/datasets/car+evaluation
'''
import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

dat = pd.read_csv("car.csv")#Using the car evaluation dataset
print(dat.head())#the first five observations
#print(np.unique(dat['class']))#The data has 4 labels
print(dat.info())
print(dat.describe())

#function to calculate the similarity score between two categorical objects
def similarity_score(x,y):
    count = 0
    for i in range(len(x)):
        if x[i]==y[i]:
            count +=1
    return count

def kmodes(data,clusters,iterations):
    index=np.random.permutation(data.index)[:clusters]#randomly selecting index values for initial k modes
    modes=[]
    labels=[]
    dis=[]
    J=[]
    d=dict()
#calculating the dissimilarity score between each categorical object from the modes and assigning label to minimum score
    for i in index:
        modes.append(np.array(data.ix[i]))#inital k modes added 
    for i in range(len(data)):
        dis=[]
        for j in range(len(modes)):
            dis.append(similarity_score(np.array(data.ix[i]),modes[j]))
        labels.append(np.argmin(dis))
    data['label']=np.array(labels)
#using frequency based method to update the weights
    for i in range(iterations):
        modes=[]
        labels=[]
        for i in np.unique(data['label']):
            modes.append(np.array(data.ix[data['label']==i,:-1].describe().ix[2,:]))
#calculating the diss score between each categorical object and updated modes and assigning new label to minimum score
        for i in range(len(data)):
            dis=[]
            for j in range(len(modes)):
                dis.append(similarity_score(np.array(data.ix[i,:-1]),modes[j]))
            labels.append(np.argmin(dis))
        data['label']=np.array(labels)
#calculating the cost function of individual cluster
    for i in range(len(modes)):
        dis=[]
        for j in range(len(data.ix[data['label']==i,:-1])):
            dis.append(similarity_score(np.array(data.ix[data['label']==i,:-1].iloc[j]),modes[i]))
        J.append(sum(dis))
    d['modes']=modes
    d['costfunction']=J
    d['totalcost']=sum(J)
    d['data']=data
    return d

dic = kmodes(dat.ix[:,:-1],4,2)#calling kmodes on the cars dataset with k equal to 4 as the original had 4 labels.
print(dic['costfunction'])#individual cost function of 4 clusters
print(dic['totalcost'])#total cost function
cldata = dic['data']#lets see how the clusterd data looks like
print(cldata.ix[cldata['label']==0,:].head())
print(cldata.ix[cldata['label']==1,:].head())
print(cldata.ix[cldata['label']==2,:].head())
print(cldata.ix[cldata['label']==3,:].head())