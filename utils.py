from mxnet import gluon
from mxnet import autograd
from mxnet import nd
import mxnet as mx
import numpy as np


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list

# from the course
class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transformer needs to process multiple examples at each time
    """
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = self.dataset[:]
        X = data[0]
        y = nd.array(data[1])
        n = X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            yield (X[i*self.batch_size:(i+1)*self.batch_size],
                   y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset)//self.batch_size




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
		# return data.astype('float32')/255, label.astype('float32')
		# data format (batch, height, weight, channel) -> (batch, channel, height, weight)
		return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

	mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=transform)
	mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=transform)

	# x, y = mnist_train[0,:]
	# print(x.shape,y.shape)

	# train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
	# test_data = gluon.data.DataLoader(mnist_test, batch_size,shuffle=False)
	train_data = DataLoader(mnist_train, batch_size, shuffle=True)
	test_data = DataLoader(mnist_test, batch_size, shuffle=False)

	# print(train_data.size)

	return (train_data, test_data)

def accuracy(output,label):
	return nd.mean(output.argmax(axis=1)==label).asscalar()


def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])


# from the course
def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n



# def evaluate_accuracy(data_iterator,net):
# 	acc = nd.array([0])
# 	n = 0.
#
# 	for data, label in data_iterator:
# 		output = net(data)
# 		acc += nd.sum(output.argmax(axis=1)==label).asscalar()
# 		n += label.size
#
# 	return acc/n





