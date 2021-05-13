"""
Bar chart of the linear discriminants of the wine dataset to find the principal component manually
"""

from wine_comon_funcs import wine_initializer, wine_matrix_init
import numpy as np
import matplotlib.pyplot as plt

mean_vectors, sc_w, sc_b = wine_matrix_init()
x_train_std, y_train, x_test_std, y_test, x, y = wine_initializer('sc_mat')
# print(sc_b)

eigen_vals, eigen_vecs=  np.linalg.eig(np.linalg.inv(sc_w).dot(sc_b))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

"""
print('Eigenvalues in decreasing order:')
for eigen_val in eigen_pairs:
    print(eigen_val[0])
"""

total = sum(eigen_vals.real)
dis_cr = [(i/total) for i in sorted(eigen_vals.real, reverse=True)]      # discriminant
cum_dis_cr = np.cumsum(dis_cr)                                           # cumulative discriminant

plt.bar(range(1, 14), dis_cr, alpha=0.5, align='center', label='individual "discriminability"')
plt.step(range(1, 14), cum_dis_cr, where='mid', label='cumulative "discriminabilty"')
plt.xlabel("Linear discriminants")
plt.ylabel('"Discriminabilty" ratio')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

"""
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)
"""

x_train_lda = x_train_std.dot(w)
colors = ['r', 'g', 'b']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_lda[y_train == l, 0], x_train_lda[y_train == l, 1], c=c, label=l, marker=m)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='upper right')
plt.show()