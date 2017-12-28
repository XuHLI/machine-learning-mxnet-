from mxnet import ndarray as nd 
from mxnet import autograd
from mxnet import gluon

import random

import matplotlib as mpl 
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

# regression with regularization

num_train = 20
num_test = 100
num_inputs = 200


true_w = nd.ones((num_inputs,1))*0.01
true_b = 0.05

X = nd.random_normal(shape=(num_train+num_test, num_inputs))
y = nd.dot(X,true_w)
y += .01*nd.random_normal(shape=y.shape)

batch_size = 1
# split into test and train sets
X_train, X_test = X[:num_train,:], X[:num_test,:]
y_train, y_test = y[:num_train], y[num_train:]

dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(dataset_train,batch_size,shuffle = True)

square_loss = gluon.loss.L2Loss()


def test(net,X,y):
	return square_loss(net(X),y).mean().asscalar()

# weight_decay*learning_rate
# how to update in trainer
# It looks like w = w-lr*grad-weight_decay*lr*w  (0<weight_decay*lr<1)
# under that condition that the penalty is lambda/2
# In this case, one can verify that this is equivalent to add L2 norm of parameters
def train(weight_decay):
	learning_rate = .002
	epochs = 10

	net = gluon.nn.Sequential()
	with net.name_scope():
		net.add(gluon.nn.Dense(1))
	net.initialize()

	trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':learning_rate, 'wd':weight_decay})
	train_loss = []
	test_loss = []

	for e in range(epochs):
		
		for data, label in data_iter_train:
			with autograd.record():
				output = net(data)
				loss = square_loss(output,label)
			loss.backward()

			trainer.step(batch_size)

		train_loss.append(test(net,X_train,y_train))
		test_loss.append(test(net,X_test,y_test))

	plt.plot(train_loss)
		# plt.plot(train_loss)
	plt.plot(test_loss)
	plt.legend(['train','test'])
	plt.show()
		# print(net[0].weight.data())
	print(net[0].weight.data())
	return ('learned w[:10]:', net[0].weight.data()[:,:10], 'learned b:', net[0].bias.data())

print(train(200))




