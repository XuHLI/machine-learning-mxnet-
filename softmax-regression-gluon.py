from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd
import utils
# from mxnet import nd

import sys
sys.path.append('..')
batch_size = 256

# Download data
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

for data, label in train_data:
	print(data.shape)
	break

net = gluon.nn.Sequential()

with net.name_scope():
	net.add(gluon.nn.Flatten())  # input 4 dimension matrix -> 2 dimension matrix
	net.add(gluon.nn.Dense(10))

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(
	net.collect_params(), 'sgd', {'learning_rate': 0.1})

learning_rate = 0.3
for epoch in range(5):
	train_acc = 0.
	train_loss = 0.

	for data, label in train_data:
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entropy(output,label)
		loss.backward()
		# utils.SGD(params,learning_rate/batch_size)

		trainer.step(batch_size)
		train_loss += nd.mean(loss).asscalar()

		train_acc += utils.accuracy(output,label)
	print(nd.max(output).asscalar())
	test_acc = utils.evaluate_accuracy(test_data,net)
	print("Epoch %d. loss: %f  train_acc:%f  test_acc: %f" %(epoch, train_loss/len(train_data),
		train_acc/len(train_data), test_acc.asscalar()))

	# print("Epoch %d. loss: %f  train_acc:%f  test_acc: %f" %(epoch, train_loss/len(train_data)/batch_size, train_acc/len(train_data), test_acc))

