from mxnet import ndarray as nd 
from mxnet import autograd

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


# split into test and train sets
X_train, X_test = X[:num_train,:], X[:num_test,:]
y_train, y_test = y[:num_train], y[num_train:]


# data iterator
batch_size = 1
def data_iter(num_examples):
	idx = list(range(num_examples))
	random.shuffle(idx)
	for i in range(0, num_examples, batch_size):
		j = nd.array(idx[i:min(i+batch_size,num_examples)])
		# return nd.take(X_train,j), nd.take(y,j)
		yield X.take(j), y.take(j)

# initialization
def get_params():
	w = nd.random_normal(shape=(num_inputs,1))*0.1
	b = nd.zeros((1,))
	for param in (w,b):
		param.attach_grad()
	return (w, b)

def net(X, lambd, w, b):
	# return nd.dot(X,w)+b+ lambd*((w**2).sum()+b**2)
	return nd.dot(X,w)+b


# loss function
def square_loss(yhat, y):
	return (yhat-y.reshape(yhat.shape))**2

# stochastic gradient descent
def SGD(params, lr):
	for param in params:
		param[:] = param - lr*param.grad

# test
def test(params,X,y):
	return square_loss(net(X,0,*params),y).mean().asscalar()

# train
def train(lambd):
	epochs  = 10
	learning_rate  = .002
	train_loss = []
	test_loss = [] 
	params = get_params()

	for e in range(epochs):
		for data, label in data_iter(num_train):
			with autograd.record():
				output = net(data,lambd,*params)
				(w,b) = params
				loss  = square_loss(output,label)+ lambd*((w**2).sum()+b**2)

			loss.backward()
			SGD(params,learning_rate)
		train_loss.append(test(params,X_train,y_train))
		test_loss.append(test(params,X_test, y_test))
	plt.plot(train_loss)
	plt.plot(test_loss)
	plt.legend(['train','test'])
	plt.show()

	return 'learned w[:10]:', params[0][:10], 'learned b:', params[1]

print(train(400))


