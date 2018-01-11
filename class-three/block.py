from mxnet import nd
from mxnet.gluon import nn

net = nn.Sequential()

with net.name_scope():
    net.add(nn.Dense(256,activation='relu'))
    net.add(nn.Dense(10))

print(net)

# nn.Block: more flexible to define a network

class MLP(nn.Block):  # a subclass of nn.Block
    def __init__(self,**kwargs): # initialize function
        super(MLP,self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)

    def forward(self,x):
        return self.dense1(nd.relu(self.dense0(x)))


net2 = MLP()
print(net2)

net2.initialize()
x = nd.random.uniform(shape=(4,10))
y = net2(x)

# print(y)

print('default prefix:', net2.dense0.name)

net3 = MLP(prefix='another_mlp_')
print('customized prefix:', net3.dense0.name)


class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    def add(self,block):
        self._children.append(block)
    def forward(self,x):
        for block in self._children:
            x = block(x)
        return x
net4 = Sequential()
with net4.name_scope():
    net4.add(nn.Dense(256))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)
print(y)

class FancyMLP(nn.Block):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,10))
    def forward(self,x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x,self.weight)+1)
        x = nd.relu(self.dense(x))
        return x

fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y = fancy_mlp(x)
print(y)


