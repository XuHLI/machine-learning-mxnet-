
from mxnet import gluon

net = gluon.nn.Sequential()

with net.name_scope():
	net.add(gluon.nn.Flatten())  # input 4 dimension -> 2 dimension
	net.add(gluon.nn.Dense(10))

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(
	net.collect_params(), 'sgd', {'learning_rate': 0.1})

