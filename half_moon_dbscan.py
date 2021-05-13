"""
DBSCAN - Density Based Spatial Clustering of Applications using Noise.
Clusters formed through this do not have to be spherical, nor do all points have to be in a cluster, i.e., noise
points can be removed
Comparing 3 unsupervised learning algorithms: DBSCAN, K-Means clustering and agglomerative clustering using
sci-kit learn's half moon dataset generator
"""

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt


x, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

f, ax = plt.subplots(2, 2, figsize=(8, 8))
k_mean = KMeans(n_clusters=2, random_state=0)
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
y_km = k_mean.fit_predict(x)
y_ac = ac.fit_predict(x)

# Training data
ax[0, 0].scatter(x[:, 0], x[:, 1], edgecolors='black')
ax[0, 0].set_title('Training data')

ax[1, 0].scatter(x[y_km == 0, 0], x[y_km == 0, 1], c='lightblue', marker='o', s=40, label='Cluster 1', edgecolors='black')
ax[1, 1].scatter(x[y_ac == 0, 0], x[y_ac == 0, 1], c='lightblue', marker='o', s=40, label='Cluster 1', edgecolors='black')
ax[1, 0].scatter(x[y_km == 1, 0], x[y_km == 1, 1], c='red', marker='s', s=40, label='Cluster 2', edgecolors='black')
ax[1, 1].scatter(x[y_ac == 1, 0], x[y_ac == 1, 1], c='red', marker='s', s=40, label='Cluster 2', edgecolors='black')

ax[1, 0].set_title('K-means clustering')
ax[1, 1].set_title('Agglomerative clustering')


# DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(x)

ax[0, 1].scatter(x[y_db == 0, 0], x[y_db == 0, 1], c='lightblue', marker='o', s=40, label='Cluster 1', edgecolors='black')
ax[0, 1].scatter(x[y_db == 1, 0], x[y_db == 1, 1], c='red', marker='s', s=40, label='Cluster 2', edgecolors='black')
ax[0, 1].set_title('DBSCAN')

plt.legend()
plt.show()

