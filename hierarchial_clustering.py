"""
Hierarchical cluster - a way to group unsupervised data, better than k-means in the way that the number of
clusters do not need to be preset.
Complete linkage - The most dissimilar(farthest) members are used to perform merges
Agglomerate - One sample is considered an individual cluster and the closest samples are merged together
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering    # sci-kit learn's implementation of the procedure

np.random.seed(123)

variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
x = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(x, columns=variables, index=labels)
# print(df)

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
# print(row_dist)

row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')  # Complete means the farthest points
print(pd.DataFrame(row_clusters, columns=['row label 1', 'row label 2', 'distance', 'number of items in cluster'],
                   index=['cluster ' + str(i) for i in range(row_clusters.shape[0])]))

# A dendogram shows the distance from the farthest cluster
row_dendogram = dendrogram(row_clusters, labels=labels)

""" To make the dendogram black:
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette(['black'])
row_dendogram = dendrogram(row_clusters, labels=labels, color_threshold=np.inf) """

plt.tight_layout()
plt.ylabel("Euclidean distance")
plt.show()

fig = plt.figure(figsize=(8, 8))
ax_dend = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')
df_row_clusters = df.ix[row_dendr['leaves'][::-1]]
ax_hmap = fig.add_axes([0.23, 0.1, 0.6, 0.6])
c_ax = ax_hmap.matshow(df_row_clusters, interpolation='nearest', cmap='hot_r')

ax_dend.set_xticks([])
ax_dend.set_yticks([])
for i in ax_dend.spines.values():
    i.set_visible(False)
fig.colorbar(c_ax)
ax_hmap.set_xticklabels([''] + list(df_row_clusters.columns))
ax_hmap.set_yticklabels([''] + list(df_row_clusters.index))
plt.show()

# Sci-kit learn hierarchical agglomerate clustering class
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(x)
print('Cluster Labels: ', labels)
