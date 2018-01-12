import OpenSSL.SSL # windows

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
x = nd.random.uniform(shape=(4,20))
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
# print(y)

class FancyMLP(nn.Block):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,20))
    def forward(self,x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x,self.weight)+1)
        x = nd.relu(self.dense(x))
        return x

fancy_mlp = FancyMLP()
fancy_mlp.initialize()

print(fancy_mlp)
y = fancy_mlp(x)
print(y.shape)

## class of nn is a sub class of nn.Block
class RecMLP(nn.Block):
    def __init__(self,**kwargs):
        super(RecMLP,self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256,activation='relu'))
            self.net.add(nn.Dense(128,activation='relu'))
            # self.dense = nn.Dense(64)
            self.dense0 = nn.Dense(15)
            self.dense1 = nn.Dense(15)
            self.denses = [self.dense0,self.dense1]
            # self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]
            # self.denses[0].weight = nd.random_uniform(shape=(256,128))
            # self.denses[1].weight = nd.random_uniform(shape=(128,256))
            # self.denses[2].weight = nd.random_uniform(shape=(64,128))
    def forward(self,x):
        x = self.net(x)
        for dense in self.denses:
            x = nd.relu(dense(x))
        return x
        # return nd.relu(self.dense(self.net(x)))

rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))

rec_mlp.initialize()

print(rec_mlp)

# y = rec_mlp(x)

# print(y)


