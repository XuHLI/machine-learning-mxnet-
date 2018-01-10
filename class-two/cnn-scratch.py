# from mxnet import nd
#
# # batch_size x channel x height x weight
# w = nd.arange(4).reshape((1,1,2,2))
# b = nd.array([1])
#
# data = nd.arange(9).reshape((1,1,3,3))
# out = nd.Convolution(data,w,b,kernel=w.shape[2:],num_filter=w.shape[1])
#
# print(w.shape)
# print('input:',data,'\n\n weight:',w,'\n\nbias:',b,'\n\noutput:',out)
#
# # kernel: size of w, stride: move horizontial and vertical
# out = nd.Convolution(data,w,b, kernel=w.shape[2:],num_filter = w.shape[1],stride=(2,2),pad=(1,1))
#
# print('input:',data,'\n\n weight:',w,'\n\nbias:',b,'\n\noutput:',out)
#
# # more channels: input multi channels. each kernel does correpsonding input and then sum them up
# w = nd.arange(8).reshape((1,2,2,2))
# data = nd.arange(18).reshape((1,2,3,3))
# out = nd.Convolution(data,w,b, kernel=w.shape[2:],num_filter = w.shape[0],stride=(2,2),pad=(1,1))
#
# print('input:',data,'\n\n weight:',w,'\n\nbias:',b,'\n\noutput:',out)
#
# # more channels: output multi channels
# w = nd.arange(16).reshape((2,2,2,2))
# data = nd.arange(18).reshape((1,2,3,3))
# b = nd.array([1,2])
# out = nd.Convolution(data,w,b, kernel=w.shape[2:],num_filter = w.shape[0])
#
# print('input:',data,'\n\n weight:',w,'\n\nbias:',b,'\n\noutput:',out)
#
#
# # pooling
#
# max_pool = nd.Pooling(data=data,pool_type="max",kernel=(2,2))
# avg_pool = nd.Pooling(data=data,pool_type="avg",kernel=(2,2))
#
# print('input:',data,'\n\nmaxpooling:',max_pool,'\n\navgpool:',avg_pool)

import sys
sys.path.append('..')
from utils import load_data_fashion_mnist

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)

import mxnet as mx
from mxnet import nd

try:
    ctx = mx.gpu()
    _ = nd.zeros((1,),ctx=ctx)
except:
    ctx = mx.cpu()

print(ctx)

weight_scale = .01
num_outputs = 10


# network: LeNet (year 89) two convolution and two fully connected

# output channels = 20, kernel = (5,5)
W1 = nd.random_normal(shape=(20,1,5,5),scale=weight_scale,ctx=ctx)
b1 = nd.zeros(W1.shape[0],ctx=ctx)

# output channels = 50, kernel = (3,3)
W2 = nd.random_normal(shape=(50,20,3,3),scale=weight_scale,ctx=ctx)
b2 = nd.zeros(W2.shape[0],ctx=ctx)

# output dim = 128
W3 = nd.random_normal(shape=(1250,128),scale=weight_scale,ctx=ctx)
b3 = nd.zeros(W3.shape[1],ctx=ctx)

# output dim = 10
W4 = nd.random_normal(shape=(W3.shape[1],10),scale=weight_scale,ctx=ctx)
b4 = nd.zeros(W4.shape[1],ctx=ctx)

params = [W1,b1, W2,b2, W3,b3, W4,b4]

for param in params:
    param.attach_grad()

# net
def net(X,verbose=False):
    X = X.as_in_context(W1.context)
    # first convolution layer
    h1_conv = nd.Convolution(data=X,weight=W1, bias = b1, kernel=W1.shape[2:],num_filter = W1.shape[0])
    h1_activation = nd.relu(h1_conv)

    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    # print(h1.shape)
    # second convolution layer
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(
        data=h2_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # print(h2.shape)

    h2 = nd.flatten(h2)

    # first layer fully connected
    h3_linear = nd.dot(h2,W3)+b3
    h3 = nd.relu(h3_linear)
    # second layer fully connected
    h4_linear = nd.dot(h3,W4)+b4

    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:',h4_linear)
    return h4_linear

# a = nd.array([[[1],[2]],[[2],[3]]])
#
# print(nd.transpose(a.astype('float32'),(2,1,0)))
#
# print(a)


# # print(train_data[:])
# for data, label in train_data:
#     # print(data.shape, label.shape)
#     net(data,verbose=True)
#     break

from mxnet import autograd as autograd
from utils import SGD, accuracy, evaluate_accuracy
from mxnet import gluon

epochs = 3
lr = 0.2
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

for epoch in range(epochs):
    train_acc = 0.
    train_loss = 0.
    for data,label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        SGD(params,lr/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)

    test_acc = evaluate_accuracy(test_data,net,ctx)

    print('Epoch %d. train loss: %f, train acc: %f, test acc: %f'%(epoch,train_loss/len(train_data),train_acc/len(train_data), test_acc))
