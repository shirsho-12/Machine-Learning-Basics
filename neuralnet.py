"""
Multi-Layer Perceptron to create a neural network with multiple hidden layers
"""

import numpy as np
import sys
from scipy.special import expit


class NeuralNetMLP:
    """ Feedforward neural network / Multi-layer perceptron classifier.
       Parameters
       ------------
       n_output : int
           Number of output units, should be equal to the
           number of unique class labels.
       n_features : int
           Number of features (dimensions) in the target dataset.
           Should be equal to the number of columns in the X array.
       n_hidden : int (default: 30)
           Number of hidden units.
       l1 : float (default: 0.0)
           Lambda value for L1-regularization.
           No regularization if l1=0.0 (default)
       l2 : float (default: 0.0)
           Lambda value for L2-regularization.
           No regularization if l2=0.0 (default)
       epochs : int (default: 500)
           Number of passes over the training set.
       eta : float (default: 0.001)
           Learning rate.
       alpha : float (default: 0.0)
           Momentum constant. Factor multiplied with the
           gradient of the previous epoch t-1 to improve
           learning speed
           w(t) := w(t) - (grad(t) + alpha*grad(t-1))
       decrease_const : float (default: 0.0)
           Decrease constant. Shrinks the learning rate
           after each epoch via eta / (1 + epoch*decrease_const)
       shuffle : bool (default: True)
           Shuffles training data every epoch if True to prevent circles.
       minibatches : int (default: 1)
           Divides training data into k minibatches for efficiency.
           Normal gradient descent learning if k=1 (default).
       random_state : int (default: None)
           Set random state for shuffling and initializing the weights.
       Attributes
       -----------
       cost_ : list
         Sum of squared errors after each epoch.
    """

    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation
                Parameters
                ------------
                y : array, shape = [n_samples]
                    Target values.
                Returns
                -----------
                onehot : array, shape = (n_labels, n_samples)
        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)
        Uses scipy.special.expit to avoid overflow
        error for very small input values z.
         expit = 1.0 / (1 + np.exp(-z))
        """
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, x, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            x_new = np.ones((x.shape[0], x.shape[1] + 1))
            x_new[:, 1:] = x
        elif how == 'row':
            x_new = np.ones((x.shape[0] + 1, x.shape[1]))
            x_new[1:, :] = x
        else:
            raise AttributeError("'how' must be 'column' or 'row'")
        return x_new

    def _feedforward(self, x, w1, w2):
        """Compute feedforward step
        Parameters
        -----------
        x : array, shape = [n_samples, n_features]
            Input layer with original features.
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.
        Returns
        ----------
        a1 : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer.
        a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer.
        z3 : array, shape = [n_output_units, n_samples]
            Net input of output layer.
        a3 : array, shape = [n_output_units, n_samples]
            Activation of output layer.
        """
        a1 = self._add_bias_unit(x, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """Compute L2 regularization cost"""
        return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        """Compute L1 regularization cost"""
        return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """Compute cost function.
        Parameters
        ----------
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer (feedforward)
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.
        Returns
        ---------
        cost : float
            Regularized cost.
        """
        term_1 = -y_enc * np.log(output)
        term_2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term_1 - term_2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost += L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Compute gradient step using backpropagation.
              Parameters
              ------------
              a1 : array, shape = [n_samples, n_features+1]
                  Input values with bias unit.
              a2 : array, shape = [n_hidden+1, n_samples]
                  Activation of hidden layer.
              a3 : array, shape = [n_output_units, n_samples]
                  Activation of output layer.
              z2 : array, shape = [n_hidden, n_samples]
                  Net input of hidden layer.
              y_enc : array, shape = (n_labels, n_samples)
                  one-hot encoded class labels.
              w1 : array, shape = [n_hidden_units, n_features]
                  Weight matrix for input layer -> hidden layer.
              w2 : array, shape = [n_output_units, n_hidden_units]
                  Weight matrix for hidden layer -> output layer.
              Returns
              ---------
              grad1 : array, shape = [n_hidden_units, n_features]
                  Gradient of the weight matrix w1.
              grad2 : array, shape = [n_output_units, n_hidden_units]
                  Gradient of the weight matrix w2.
        """
        # backpropagation
        sigma_3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma_2 = w2.T.dot(sigma_3) * self._sigmoid_gradient(z2)
        sigma_2 = sigma_2[1:, :]
        grad_1 = sigma_2.dot(a1)
        grad_2 = sigma_3.dot(a2.T)

        # regularize
        grad_1[:, 1:] += self.l2 * w1[:, 1:]
        grad_1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad_2[:, 1:] += self.l2 * w2[:, 1:]
        grad_2[:, 1:] += self.l1 * np.sign(w2[:, 1:])
        
        return grad_1, grad_2

    def predict(self, x):
        """Predict class labels
               Parameters:
                x : array, shape = [n_samples, n_features]
                   Input layer with original features.
               Returns:
                y_pred : array, shape = [n_samples]
                    Predicted class labels.
        """
        if len(x.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')
        a1, z2, a2, z3, a3 = self._feedforward(x, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, x, y, print_progress=False):
        """ Learn weights from training data.
                Parameters:
                    x : array, shape = [n_samples, n_features]
                        Input layer with original features.
                    y : array, shape = [n_samples]
                        Target class labels.
                    print_progress : bool (default: False)
                        Prints progress as the number of epochs to stderr.
                Returns:
                    self
        """
        self.cost_ = []
        x_data, y_data = x.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: {0}/{1}'.format(i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                index = np.random.permutation(y_data.shape[0])
                x_data, y_enc = x_data[index], y_enc[:, index]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for index in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(x_data[index], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, index], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)

                grad_1, grad_2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2,
                                                    y_enc=y_enc[:, index], w1=self.w1, w2=self.w2)
                delta_w1, delta_w2 = self.eta * grad_1, self.eta * grad_2
                self.w1 -= (delta_w1 + self.alpha * delta_w1_prev)
                self.w2 -= (delta_w2 + self.alpha * delta_w2_prev)
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
            
        return self


class MLPGradientCheck(NeuralNetMLP):

    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        super().__init__(n_output, n_features, n_hidden, l1, l2, epochs, eta,
                        alpha, decrease_const, shuffle, minibatches)
        np.random.seed(random_state)

    def _gradient_checking(self, x, y_enc, w1, w2, epsilon, grad_1, grad_2):
        """Apply gradient checking (for debugging only)
        Returns:
              relative error: float
                Relative error between the numerically approximated gradients and the
                backpropagated gradients.
        """
        # Backtracked gradients
        # Input layer error
        num_grad_1 = np.zeros(np.shape(w1))
        epsilon_ar_y1 = np.zeros(np.shape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_ar_y1[i, j] = epsilon
                # Calculating the symmetric/centered difference quotient
                a1, z2, a2, z3, a3 = self._feedforward(x, w1 - epsilon_ar_y1, w2)
                cost_1 = self._get_cost(y_enc, a3, w1 - epsilon_ar_y1, w2)

                a1, z2, a2, z3, a3 = self._feedforward(x, w1 + epsilon_ar_y1, w2)
                cost_2 = self._get_cost(y_enc, a3, w1 + epsilon_ar_y1, w2)

                num_grad_1[i, j] = (cost_2 - cost_1) / (2 * epsilon)
                epsilon_ar_y1[i, j] = 0

        # Hidden layer error
        num_grad_2 = np.zeros(np.shape(w2))
        epsilon_ar_y2 = np.zeros(np.shape(w2))
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_ar_y2[i, j] = epsilon
                # Calculating the symmetric/centered difference quotient
                a1, z2, a2, z3, a3 = self._feedforward(x, w1, w2 - epsilon_ar_y2)
                cost_1 = self._get_cost(y_enc, a3, w1, w2 - epsilon_ar_y2)

                a1, z2, a2, z3, a3 = self._feedforward(x, w1, w2 + epsilon_ar_y2)
                cost_2 = self._get_cost(y_enc, a3, w1, w2 + epsilon_ar_y2)

                num_grad_2[i, j] = (cost_2 - cost_1) / (2 * epsilon)
                epsilon_ar_y2[i, j] = 0

        num_grad = np.hstack((num_grad_1.flatten(), num_grad_2.flatten()))
        grad = np.hstack((grad_1.flatten(), grad_2.flatten()))
        norm_1 = np.linalg.norm(num_grad - grad)
        norm_2 = np.linalg.norm(num_grad)
        norm_3 = np.linalg.norm(grad)
        relative_error = norm_1 / (norm_2 + norm_3)
        return relative_error

    def fit(self, x, y, print_progress=False):
        """ Learn weights from training data.
                Parameters
                -----------
                X : array, shape = [n_samples, n_features]
                    Input layer with original features.
                y : array, shape = [n_samples]
                    Target class labels.
                print_progress : bool (default: False)
                    Prints progress as the number of epochs
                    to stderr.
                Returns:
                ----------
                self
        """
        self.cost_ = []
        x_data, y_data = x.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: {0}/{1}'.format(i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                index = np.random.permutation(y_data.shape[0])
                x_data, y_enc = x_data[index], y_enc[:, index]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for index in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(x_data[index], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, index], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)

                # Compute gradients via backpropagation
                grad_1, grad_2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2,
                                                    y_enc=y_enc[:, index], w1=self.w1, w2=self.w2)

                # Start gradient checking
                grad_diff = self._gradient_checking(x=x[index], y_enc=y_enc[:, index], w1=self.w1,
                                                    w2=self.w2, epsilon=1e-5,
                                                    grad_1=grad_1, grad_2=grad_2)
                if grad_diff <= 1e-7:
                    print('OK:', grad_diff)
                elif grad_diff < 1e-4:
                    print('Warning:', grad_diff)
                else:
                    print("PROBLEM:", grad_diff)
                # Gradient checking ends

                # update weights: alpha * delta_w_prev for momentum learning
                delta_w1, delta_w2 = self.eta * grad_1, self.eta * grad_2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self
