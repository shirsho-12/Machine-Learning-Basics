"""
Using some of the various array functions for Theano-NumPy functionality
"""

import theano
from theano import tensor as T
import numpy as np

"""Arrays and Python lists"""
# init
x = T.fmatrix(name='x')
x_sum = T.sum(x, axis=0)

# compile
calc_sum = theano.function(inputs=[x], outputs=x_sum)

# exec (Python list)
arr = [[1, 2, 3], [1, 2, 3]]
print('Column Sum:', calc_sum(arr))

# exec (NumPy array)
theano.config.floatX = 'float32'
arr = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
print('Column Sum:', calc_sum(arr))

"""Using shared"""
# init
x = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))
# allows array to be spread over CPU, better memory management
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[x], updates=update, outputs=z)

# exec
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
for i in range(5):
    print('z{0}: {1}'.format(i, net_input(data)))

"""Using givens"""
# init
x = T.fmatrix('x')
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[], updates=update, givens={x: data}, outputs=z)
# givens reduces the number of transfer from RAM over CPUs to GPUs to speed up algorithms that use
# shared variables

# exec
for i in range(5):
    print('z{0}: {1}'.format(i, net_input()))
