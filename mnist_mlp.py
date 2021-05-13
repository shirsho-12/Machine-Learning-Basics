"""
Multi-layer Perceptron- Deep learning algorithm implemented on the MNIST handwritten digit database
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from net2 import NeuralNetMLP, MLPGradientCheck
# from sklearn.neural_network import MLPClassifier


# Loading the MNIST dataset of handwritten digits 0 to 9
def load_mnist(path, kind='train'):
    """Load MNIST data from path"""
    labels_path = os.path.join(path, '{0}-labels.idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{0}-images.idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbl_path:
        magic, n = struct.unpack('>II', lbl_path.read(8))
        labels = np.fromfile(lbl_path, dtype=np.uint8)

    with open(images_path, 'rb') as img_path:
        magic, num, rows, cols = struct.unpack(">IIII", img_path.read(16))
        images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


x_train, y_train = load_mnist('mnist', kind='train')
print('Rows:', x_train.shape[0], ' Columns:', x_train.shape[1])
x_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows:', x_test.shape[0], ' Columns:', x_test.shape[1])


def num_plots():
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True)
    ax = ax.flatten()
    for i in range(10):
        img = x_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True)
    ax = ax.flatten()
    for i in range(25):
        img = x_train[y_train == 7][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()


# num_plots()
"""
To save the ubyte files as csv the NumPy function savetxt can be used as follows
np.savetxt('train_img.csv', x_train, fmt='%i', delimiter=',')
However, file sizes are larger and load times are greater.
"""


# 784-50-10 MLP, neural network with 784 input units, 50 hidden units and 10 output units
def n_net():
    nn = NeuralNetMLP(n_output=10, n_features=x_train.shape[1], n_hidden=50, l2=0.1, l1=0.0,
                      epochs=1000, eta=0.001, alpha=0.001, decrease_const=1e-5,
                      shuffle=True, minibatches=50, random_state=1)

    # sci-kit learn MLP classifier- remove print_progress parameter from fit and comment out the plots
    # nn = MLPClassifier(solver='lbfgs', alpha=0.001, random_state=1, activation='logistic', verbose=1)

    nn.fit(x_train, y_train, print_progress=True)

    y_train_pred = nn.predict(x_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / x_train.shape[0]
    print("Training accuracy: {0:.2f}%".format(acc * 100))

    y_test_pred = nn.predict(x_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / x_test.shape[0]
    print("Test accuracy: {0:.2f}%".format(acc * 100))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(range(len(nn.cost_)), nn.cost_)
    ax1.set_ylabel('Cost')
    ax1.set_xlabel('Epochs * 50')
    ax1.set_title('Before averaging')

    batches = np.array_split(range(len(nn.cost_)), 1000)
    cost_arr_y = np.array(nn.cost_)
    cost_avgs = [np.mean(cost_arr_y[i]) for i in batches]
    ax2.plot(range(len(cost_avgs)), cost_avgs, color='red')
    ax2.set_ylim([0, 2000])
    ax2.set_ylabel('Cost')
    ax2.set_xlabel('Epochs')
    ax2.set_title('After averaging over minibatch intervals')

    plt.tight_layout()
    plt.show()

    misclassified_img = x_test[y_test != y_test_pred][:25]
    correct_label = y_test[y_test != y_test_pred][:25]
    misclassified_label = y_test_pred[y_test != y_train_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(5):
        img = misclassified_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('{0}) T: {1} P:{2}'.format(i + 1,correct_label[i], misclassified_label[i]))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()


nn_check = MLPGradientCheck(n_output=10, n_features=x_train.shape[1], n_hidden=10, l2=0.0, l1=0.0,
                            epochs=10, eta=0.001, alpha=0.0, decrease_const=0.0,
                            minibatches=1, random_state=1)

nn_check.fit(x_train[:5], y_train[:5], print_progress=False)
