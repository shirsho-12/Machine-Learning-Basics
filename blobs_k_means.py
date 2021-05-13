"""
Unsupervised learning. Finding clusters of data using the K-Means algorithm and comparing the results by different
number of clusters selected using the elbow method.
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random

x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5,
                  shuffle=True, random_state=random.seed())
# plt.scatter(x[:, 0], x[:, 1], c='white', marker='o', edgecolors='black', s=50)
plt.grid()

k_mean = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=0)
y_km = k_mean.fit_predict(x)
print("Distortion: ", k_mean.inertia_)

plt.scatter(x[y_km == 0, 0], x[y_km == 0, 1], s=50, c='lightgreen', marker='s',
            label='Cluster 1', edgecolors='black')
plt.scatter(x[y_km == 1, 0], x[y_km == 1, 1], s=50, c='orange', marker='o',
            label='Cluster 2', edgecolors='black')
plt.scatter(x[y_km == 2, 0], x[y_km == 2, 1], s=50, c='lightblue', marker='v',
            label='Cluster 3',  edgecolors='black')
plt.scatter(k_mean.cluster_centers_[:, 0], k_mean.cluster_centers_[:, 1], s=250,
            marker='*', c='red', label='Centroids',  edgecolors='black')
plt.legend()
plt.show()


# Elbow method to compare distortions/inertia
def elbow_plot():
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=0)
        km.fit(x)
        distortions.append(km.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortions')
    plt.show()


"""
The point after which the decrease in distortions shows a linear relationship with the number of clusters
is the optimal number of clusters.
"""
# elbow_plot()

# Silhouette analysis- calculating the silhouette coefficient to evaluate the quality of clustering in a dataset
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(x, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
y_ticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    y_ticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

# A mean close to 0 means bad clustering, while near-uniform cluster bar heights and widths mean optimal clustering
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(y_ticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.show()
