import pip
for pkg in ['mxnet','mxnet-cu75','mxnet-cu80']:
    pip.main(['show',pkg])

import mxnet as mx
print([mx.cpu(), mx.gpu(),mx.gpu(1)])

# default context: CPU
from mxnet import nd
x = nd.array([1,2,3])
print(x)

a = nd.array([1,2,3], ctx=mx.gpu()) # set context to be GPU
b = nd.zeros((3,2), ctx=mx.gpu())
c = nd.random_normal(shape=(2,3),ctx=mx.gpu())

print((a,b,c))

# check how many number of GPUs you have. If gpu does not exist, there the error comes up
# import sys
#
# try:
#     print(nd.array([1,2,3],ctx=mx.gpu(2)))
# except mx.MXNetError as err:
#     sys.stderr.write(str(err))

# from cpu to gpu
y = x.copyto(mx.gpu()) # new create always
z = x.as_in_context(mx.gpu()) # if in gpu already, there it is itself, no copy
print((y,z))

yy = y.as_in_context(mx.gpu())
zz = z.copyto(mx.gpu())

print((yy is y, zz is z))

# y, z are in gpu, then the computation is on gpu
print(nd.exp(z+2)*y)

# x in cpu, y in gpu -> error
# import sys
# try:
#     print(x+y)
# except mx.MXNetError as err:
#     sys.stderr.write(str(err))


print(y) # in gpu, ndarray supports gpu
print(y.asnumpy()) # asnumpy does not support gpu, in cpu
print(y.sum().asscalar()) # in cpu


# gluon in context
from mxnet import gluon
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

net.initialize(ctx=mx.gpu())

data = nd.random_uniform(shape=[3,2],ctx=mx.gpu())
print(net(data))

print(net[0].weight.data())