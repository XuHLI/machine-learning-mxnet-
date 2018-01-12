# import OpenSSL.SSL
from mxnet.gluon import nn


# alex net 

net = nn.Sequential()

with net.name_scope():
	# first phase
	net.add(nn.Conv2D(
		channels=96, kernel_size=11, strides=4, activation='relu'))
	net.add(nn.MaxPool2D(pool_size=3, strides=2))

	# second phase
	net.add(nn.Conv2D(
		channels=256, kernel_size=5, padding=2, activation='relu'))
	net.add(nn.MaxPool2D(pool_size=3, strides=2))

	# third phase
	net.add(nn.Conv2D(
		channels=384, kernel_size=3, padding=1, activation='relu'))
	net.add(nn.Conv2D(
		channels=384, kernel_size=3, padding=1, activation='relu'))
	net.add(nn.Conv2D(
		channels=256, kernel_size=3, padding=1, activation='relu'))
	net.add(nn.MaxPool2D(pool_size=3, strides=2))


	# fourth phase
	net.add(nn.Flatten())
	net.add(nn.Dense(4096,activation='relu'))
	net.add(nn.Dropout(.5))

	# fifth phase
	net.add(nn.Dense(4096,activation='relu'))
	net.add(nn.Dropout(.5))

	# sixth phase
	net.add(nn.Dense(10))


# use FashionMNIST dataset instead of Imagenet
import sys
sys.path.append('..')

import utils
from mxnet import image 

def transform(data,label):
	# 28 x 28 -> 224 x 224
	data = image.imresize(data,224,224)
	return utils.transform_mnist(data,label)

batch_size = 64
resize = 224
train_data, test_data = utils.load_data_fashion_mnist(batch_size,resize)

# train
from mxnet import gluon, autograd, nd, init
ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(
	net.collect_params(), 'sgd',{'learning_rate':0.01})


for epoch in range(3):
	train_loss = 0.
	train_acc = 0.
	for data, label in train_data:
		label = label.as_in_context(ctx)
		with autograd.record():
			output = net(data.as_in_context(ctx))
			loss = softmax_cross_entropy(output,label)

		loss.backward()
		trainer.step(batch_size)

		train_loss +=nd.mean(loss).asscalar()
		train_acc += utils.accuracy(output,label)

	test_acc = utils.evaluate_accuracy(test_data,net,ctx)
	print("Epoch %d. train loss: %f, train acc: %f, test acc: %f"%(epoch,train_loss/len(train_data), train_acc/len(train_data),test_acc))



