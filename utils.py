from mxnet import gluon
from mxnet import autograd
from mxnet import nd
import mxnet as mx



# Stochastic gradient decent: 
# params: storage of parameters(a vector)
# lr: learning rate
def SGD(params,lr):
	for param in params:
		param[:] = param - lr*param.grad


# load data 
def load_data_fashion_mnist(batch_size,resize=None, root="~/.mxnet/datasets/fashion-mnist"):
	'''Download the fashion mnist dataset, data format (batch, height, weight, channel)'''
	def transform(data,label):
		if resize:
			n = data.shape[0]
			new_data = nd.zeros((n,resize,resize, data.shape[3]))
			for i in range(n):
				new_data[i] = image.imresize(data[i],resize, resize)
			data = new_data
		return data.astype('float32')/255, label.astype('float32')
		# data format (batch, height, weight, channel) -> (batch, channel, height, weight)
		# return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32') 

	mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=transform)
	mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=transform)

	train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
	test_data = gluon.data.DataLoader(mnist_test, batch_size,shuffle=False)

	return (train_data, test_data)

def accuracy(output,label):
	return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator,net):
	acc = nd.array([0])
	n = 0.

	for data, label in data_iterator:
		output = net(data)
		acc += nd.sum(output.argmax(axis=1)==label).asscalar()
		n += label.size

	return acc/n





