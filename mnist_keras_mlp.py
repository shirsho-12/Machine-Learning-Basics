"""
Using Tensorflow-Keras to train a convolutional neural network (CNN) of size 784-50-50-10 using
stochastic gradient descent (SGD)

To set the flag and gpu settings:
statement in terminal window(Bash):
   set THEANO_FLAGS="mode=FAST_RUN"  & set THEANO_FLAGS="device=gpu"
    & set THEANO_FLAGS="floatX=float32" & python theanogpu_example.py

This code does not run in this project due to the lack of an Anaconda configuration.
Go to project ML-TensorFLow(Keras) folder: ML2 to run this instead
This file, when executed will produce an error stating that.
"""
import sys
try:
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.optimizers import SGD
    import theano
    import numpy as np
    import struct
    import os

    theano.config.floatX='float32'
    np.random.seed(1)


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

    x_train = x_train.astype(theano.config.floatX)
    x_test = x_test.astype(theano.config.floatX)

    print("First 3 labels: ", y_train[:3])
    y_train_ohe = np_utils.to_categorical(y_train)
    print('\nFirst 3 labels (one-hot): \n',y_train_ohe[:3])


    model = Sequential()
    # input layer
    model.add(Dense(input_dim=x_train.shape[1], units=50, kernel_initializer='uniform', activation='tanh'))
    # hidden layer
    model.add(Dense(input_dim=50, units=50, kernel_initializer='uniform', activation='tanh'))
    # output layer
    model.add(Dense(input_dim=50, units=y_train_ohe.shape[1], kernel_initializer='uniform', activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train_ohe, epochs=50, batch_size=300, verbose=1,
              validation_split=0.1)

    y_train_pred = model.predict_classes(x_train, verbose=0)
    print("First 3 predictions: ",y_train_pred[:3])

    train_acc = np.sum(y_train==y_train_pred, axis=0) / x_train.shape[0]
    print("Training accuracy: {0:.2f}".format(train_acc * 100))
    y_test_pred = model.predict_classes(x_test, verbose=0)
    test_acc = np.sum(y_test==y_test_pred, axis=0) / x_test.shape[0]
    print("Test accuracy: {0:.2f}".format(test_acc * 100))

except ModuleNotFoundError:
    print("ERROR\n"
          "- - - \n "
          "Wrong project: go to project name: 'ML-TensorFlow(Keras) folder name: ML2 and run the file through"
          " cmd(Bash) - Instructions given in file comments.", file=sys.stderr)
