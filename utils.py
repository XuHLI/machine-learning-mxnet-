from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon import nn


# Stochastic gradient decent: 
# params: storage of parameters(a vector)
# lr: learning rate
def SGD(params,lr):
	for param in params:
		param[:] = param - lr*param.grad