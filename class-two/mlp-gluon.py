from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd 

net = gluon.nn.Sequential()
with net.name_scope():
	net.add(gluon.nn.Flatten())
	net.add(gluon.nn.Dense(256,activation='relu')) # hidden layer with 256 nodes
	net.add(gluon.nn.Dense(128,activation='relu')) # can add more layers
	net.add(gluon.nn.Dense(10))
print(net)
net.initialize()

import sys
sys.path.append('..')
import utils

batch_size = 256

train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# loss function
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# train
trainer = gluon.Trainer(
	net.collect_params(), 'sgd', {'learning_rate': .5})


for epoch in range(5):
	train_acc = 0.
	train_loss = 0. 
	for data, label in train_data:
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entropy(output, label)

		loss.backward()
		trainer.step(batch_size)
		train_acc += utils.accuracy(output,label)
		train_loss += nd.mean(loss).asscalar()

	test_acc = utils.evaluate_accuracy(test_data,net).asscalar()

	print("Epoch: %d, train loss: %f, train accuracy: %f, test accuracy: %f" %(epoch, 
		train_loss/len(train_data), train_acc/len(train_data), test_acc))


