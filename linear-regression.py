from mxnet import ndarray as nd 
from mxnet import autograd


# generate data
num_inputs = 2
num_examples = 1000

true_w = [2 , -3.4]
true_b = 4.2


X = nd.random_normal(shape=(num_examples,num_inputs))
y = true_w[0]*X[:,0]+true_w[1]*X[:,1]+true_b
y += .01*nd.random_normal(shape=y.shape)  # add noise

# print(X[0:10],y[0:10])

# read data
import random
batch_size = 10 # read number of data each time
def data_iter():
	idx = list(range(num_examples)) # generate a random index
	random.shuffle(idx) # a order
	for i in range(0,num_examples,batch_size):
		j = nd.array(idx[i:min(i+batch_size,num_examples)])
		yield nd.take(X,j), nd.take(y,j)


for data, label in data_iter():
	break


# parameters: initialization
w = nd.random_normal(shape=(num_inputs,1))
b = nd.zeros((1,))
params = [w, b]

for param in params:
	param.attach_grad()

# model
def net(X):
	return nd.dot(X,w)+b

# loss function
def square_loss(yhat,y):
	return (yhat-y.reshape(yhat.shape))**2

#  lr --- learning rate
def SGD(params, lr):
	for param in params:
		param[:] = param - lr*param.grad

# train
epochs = 10 # go through data 5 times
learning_rate = .05
for e in range(epochs):
	total_loss = 0
	for data, label in data_iter():
		with autograd.record():
			output = net(data)
			loss = square_loss(output,label)
		loss.backward() # add gradient to parameter: param.grad
		SGD(params,learning_rate) # update the parameter

		total_loss += nd.sum(loss).asscalar()
	print("Epoch %d, average loss: %f" %(e,total_loss/num_examples))

print(true_b,b)
