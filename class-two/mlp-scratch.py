import sys
sys.path.append('..')

import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)


from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd

num_inputs = 28*28
num_outputs = 10

num_hidden = 200
weight_scale = .01

# num_hidden_1 = 200
# initialization
# first layer: input layer to hidden layer
W1 = nd.random_normal(shape=(num_inputs,num_hidden), scale = weight_scale)
b1 = nd.zeros(num_hidden)

# second layer: hidden layer to output layer
W2 = nd.random_normal(shape=(num_hidden,num_outputs), scale = weight_scale)
b2 = nd.zeros(num_outputs)


params = [W1,b1, W2, b2]

# # hidden layer 0 -> hidden layer 1
# W2 = nd.random_normal(shape=(num_hidden,num_hidden_1), scale = weight_scale)
# b2 = nd.zeros(num_hidden_1)
#
# # hidden layer 1 -> output
# W3 = nd.random_normal(shape=(num_hidden_1,num_outputs), scale = weight_scale)
# b3 = nd.zeros(num_outputs)
#
# params = [W1, b1, W2, b2, W3, b3]
# print(params)
# attach gradient for later use
for param in params:
    param.attach_grad()

# nonlinear transformation: relu(x) = max(x,0)  easy implementment convenient to compute
def relu(X):
    return nd.maximum(X,0)

# model
def net(X):
    X = X.reshape((-1,num_inputs)) # -1 here is batch_size
    h1 = relu(nd.dot(X,W1)+b1)
    # h2 = relu(nd.dot(h1,W2)+b2)
    output = nd.dot(h1,W2)+b2
    return output

# softmax function: for stability
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


# learning rate
learning_rate = .5

for epoch in range(5):
    train_acc = 0.
    train_loss = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()

        utils.SGD(params,learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    test_acc = utils.evaluate_accuracy(test_data,net)

    print("Epoch:%d, loss: %f, train_acc: %f, test_acc:%f"%(epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc.asscalar()))