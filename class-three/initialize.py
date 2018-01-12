import OpenSSL.SSL # windows
from mxnet import nd
from mxnet.gluon import nn

def get_net():
	net  = nn.Sequential()
	with net.name_scope():
		net.add(nn.Dense(4,activation='relu'))
		net.add(nn.Dense(2))

	return net



x = nd.random.uniform(shape=(3,5))

# no initialization
import sys
try:
	net = get_net()
	net.initialize()
	print(net)

except RuntimeError as err:
	sys.stderr.write(str(err))

y = net(x)
print(y)
w = net[0].weight
b = net[0].bias


print(w.grad())

print('name: ', net[0].name)
net[1].bias
# print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)

params = net.collect_params()

# print(params)

from mxnet import init

params.initialize(init=init.One(),force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())

net = get_net()
# print(net.collect_params())

## How to read and serialization

from mxnet import nd

x = nd.ones(3)
y = nd.zeros(4)

filename = "../data/test1.params"
nd.save(filename,[x,y])

a,b = nd.load(filename)
print(a,b)

# save a dictionary
mydict = {"x":x,"y":y}
filename = "../data/test2.params"
nd.save(filename,mydict)

c = nd.load(filename)
print(c)

import mxnet as mx
# gluon
net = get_net()
net.initialize()
x = nd.random.uniform(shape=(3,5))
print(net(x))
###############################################
#     save_params and load_params 
###############################################

# net.save_params: save parameter
filename = "../data/mlp.params"
net.save_params(filename)

# net.load_params:  load parameter
net2 = get_net()
net2.load_params(filename,mx.cpu())

print(net2(x))



#########################################################
#     custom-layer
#########################################################

from mxnet.gluon import nn

class CenteredLayer(nn.Block):
	def __init__(self,**kwargs):
		super(CenteredLayer,self).__init__(**kwargs)

	def forward(self,x):
		return x-x.mean()

layer = CenteredLayer()
print(layer(nd.array([1,2,3,4,5])))

net = nn.Sequential()
with net.name_scope():
	net.add(nn.Dense(128))
	net.add(nn.Dense(10))
	net.add(CenteredLayer())

net.initialize() # net has parameters so need initialization
y = net(nd.random.uniform(shape=(3,2)))

print(y)
print(y.mean())


## with parameters
from mxnet import gluon
my_param = gluon.Parameter("exciting_parameter_yay", shape=(3,3)) # prefix, shape

my_param.initialize()

print(my_param.data(), my_param.grad())


# another way
pd = gluon.ParameterDict(prefix="block1_")
pd.get("exciting_parameter_yay",shape=(3,3))
# print(pd)

class MyDense(nn.Block):
	def __init__(self,units,in_units,**kwargs):
		super(MyDense,self).__init__(**kwargs)
		with self.name_scope():
			self.weight = self.params.get(
				'weight', shape=(in_units,units))
			self.bias = self.params.get('bias',shape=(units,))

	def forward(self,x):
		linear = nd.dot(x,self.weight.data())+self.bias.data()
		return nd.relu(linear)

dense = MyDense(5,in_units=10,prefix='o_my_dense_')
# print(dense.params)
dense.initialize()
dense(nd.random.uniform(shape=(10,10)))

net = nn.Sequential()
with net.name_scope():
	net.add(MyDense(32,in_units=64))
	net.add(MyDense(2,in_units=32))

net.initialize()
y = net(nd.random.uniform(shape=(2,64)))
print(y)