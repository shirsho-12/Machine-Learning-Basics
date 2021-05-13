"""
A basic Theano function
Theano settings configuration
"""

import theano
from theano import tensor as T

# initialization
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# compilation
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# execution
print('Net input: {0:.2f}'.format(net_input(2.0, 1.0, 0.5)))

print(theano.config.floatX, theano.config.device)
theano.config.floatX = 'float32'
# export THEANO_FLAGS=floatX=float32 does the same thing, but only in the Terminal window
# changing he device from cpu to gpu is only possible in the terminal with similar code
print(theano.config.floatX)
