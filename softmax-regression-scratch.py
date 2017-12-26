from mxnet import gluon
from mxnet import ndarray as nd 

def transform(data,label):
	return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train=True,transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False,transform=transform)

data, label = mnist_train[0]

print('example shape: ', data.shape, 'label:', label)


# import matplotlib.pyplot as plt 

# def show_images(images):
# 	n = images.shape[0]
# 	_, figs = plt.subplots(1,n,figsize=(15,15))
# 	for i in range(n):
# 		figs[i].imshow(images[i].reshape((28,28)).asnumpy())
# 		figs[i].axes.get_xaxis().set_visible(False)
# 		figs[i].axes.get_yaxis().set_visible(False)

# 	plt.show()

# def get_text_labels(label):
# 	text_labels = [
# 	  't-shirt', 'trouser', 'pullover','dress','coat', 
# 	  'sandal', 'shirt','sneaker','bag','ankle boot'
# 	  ]
# 	return [text_labels[int(i)] for i in label]

# data, label = mnist_train[0:9]
# show_images(data)
# print(get_text_labels(label))

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test,batch_size,shuffle=False)

# a different attainment


num_examples = len(mnist_train)

data,label = mnist_train[:]
print(data.shape,label.shape)

test = data

label = nd.array(label)

print(nd.take(test,nd.array([1,2])).shape)
print(nd.take(label[:],nd.array([1,2])))

for data,label in train_data:
	print(data.shape, label.shape)
	break


# print(num_examples)
# for data, label in train_data:
# 	print(data.shape, label)
# 	break
import random
def data_iter():
	idx = list(range(num_examples))
	random.shuffle(idx)
	data, label = mnist_train[:]
	for i in range(0,num_examples, batch_size):
		j = nd.array(idx[i:i+min(batch_size,num_examples)])
		yield nd.take(data,j), nd.take(nd.array(label),j)
# nd.take(data,j), 
# data, label = data_iter()

for data, label in data_iter():
	# print(label)
	break


# Define model
num_inputs = 784
num_outputs = 10

W = nd.random_normal(shape=(num_inputs,num_outputs))
b = nd.random_normal(shape=num_outputs)

params = [W, b]

for param in params:
	param.attach_grad()


# Define model
# from mxnet import nd


# softmax: make a general result to a probability one
def softmax(X):
	exp = nd.exp(X)
	partition = exp.sum(axis=1, keepdims=True)
	return exp/partition #  (nrows,1) 


# X = nd.random_normal(shape=(2,5))

# X_prob = softmax(X)

# print(X)
# print(X_prob)

def net(X):
	return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b)  # -1:  batch_size

def cross_entropy(yhat,y):
	return -nd.pick(nd.log(yhat),y)

def accuracy(output,label):
	return nd.mean(output.argmax(axis=1)==label).asscalar()


# mainly use for test process
def evaluate_accuracy(data_iterator,net):
	acc = 0.
	for data, label in data_iterator:
		output = net(data)
		acc += accuracy(output,label)

	return acc / len(data_iterator)

print(evaluate_accuracy(test_data,net))


# print(params)
# for param in params:
# 	print(param.grad)
# 	break


import sys
sys.path.append('..')
from utils import SGD
from mxnet import autograd

learning_rate = .3


for epoch in range(4):
	train_loss = 0.
	train_acc = 0. 
	for data, label in train_data:
		with autograd.record():
			output = net(data)
			loss  = cross_entropy(output,label)
		loss.backward()
		

		SGD(params, learning_rate/batch_size)

		train_loss += nd.mean(loss).asscalar()
		train_acc += accuracy(output,label)

	print(nd.max(output).asscalar())

	test_acc = evaluate_accuracy(test_data,net)

	print("Epoch %d .Loss: %f, Train acc %f, Test acc %f" %(epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))

# a new version using data_iter()
# for epoch in range(2):
# 	train_loss = 0.
# 	train_acc = 0.
# 	n = 0
# 	for data, label in data_iter():
# 		with autograd.record():
# 			output = net(data)
# 			loss = cross_entropy(output,label)
# 		loss.backward()
# 		SGD(params,learning_rate/batch_size)

# 		train_loss += nd.mean(loss).asscalar()
# 		train_acc += accuracy(output,label)
# 		n += 1

# 	test_acc = evaluate_accuracy(test_data,net)

# 	print("Epoch %d train loss: %f train acc: %f, test acc: %f" %(epoch, train_loss/n,train_acc/n, test_acc))
