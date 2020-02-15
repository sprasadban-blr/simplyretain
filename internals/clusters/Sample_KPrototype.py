#!/usr/bin/env python
import numpy as np
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
#Data points with their publisher name,category score, category name, place name
syms = np.genfromtxt('travel.csv', dtype=str, delimiter=',')[:, 1]
X = np.genfromtxt('travel.csv', dtype=object, delimiter=',')[:, 2:]
X[:, 0] = X[:, 0].astype(float)
kproto = KPrototypes(n_clusters=3, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[1, 2])
# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)
for s, c in zip(syms, clusters):
    print("Result: {}, cluster:{}".format(s, c))
# Plot the results
for i in set(kproto.labels_):
    index = kproto.labels_ == i
    x = X[index, 0]
    y = X[index, 1]
    plt.plot(x, y, 'o')
    plt.suptitle('Data points categorized with category score', fontsize=18)
    plt.xlabel('Category Score', fontsize=16)
    plt.ylabel('Category Type', fontsize=16)
plt.show()
# Clustered result
fig1, ax3 = plt.subplots()
scatter = ax3.scatter(syms, clusters, c=clusters, s=50)
ax3.set_xlabel('Data points')
ax3.set_ylabel('Cluster')
plt.colorbar(scatter)
ax3.set_title('Data points classifed according to known centers')
plt.show()
result = zip(syms, kproto.labels_)
sortedR = sorted(result, key=lambda x: x[1])
print(sortedR)