from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    with out.name_scope():
        for _ in range(num_convs):
            out.add(nn.Conv2D(channels=channels, kernel_size=3,
                              padding =1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2,strides=2))
    return out


from mxnet import nd

blk = vgg_block(2,128)
blk.initialize()

x = nd.random.uniform(shape=(2,3,16,16)) # batch_size x channels x height x weight
y = blk(x)

print(y.shape)

def vgg_stack(architecture):
    out = nn.Sequential()
    with out.name_scope():
        for (num_convs,channels) in architecture:
            out.add(vgg_block(num_convs,channels))

    return out

num_output = 10
architecture = ((1,64),(1,128),(2,256),(2,512),(2,512))

net = nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(4096,activation='relu'))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096,activation='relu'))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_output))

print(net)

import sys
sys.path.append('..')

import utils
from mxnet import image

def transform(data,label):
	# 28 x 28 -> 96 x 96
	data = image.imresize(data,224,224)
	return utils.transform_mnist(data,label)

batch_size = 64
resize = 96
train_data, test_data = utils.load_data_fashion_mnist(batch_size,resize)


# train
from mxnet import gluon, autograd, nd, init
ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(
	net.collect_params(), 'sgd',{'learning_rate': 0.01})


for epoch in range(1):
	train_loss = 0.
	train_acc = 0.
	for data, label in train_data:
		label = label.as_in_context(ctx)
		with autograd.record():
			output = net(data.as_in_context(ctx))
			loss = softmax_cross_entropy(output,label)

		loss.backward()
		trainer.step(batch_size)

		train_loss +=nd.mean(loss).asscalar()
		train_acc += utils.accuracy(output,label)

	test_acc = utils.evaluate_accuracy(test_data,net,ctx)
	print("Epoch %d. train loss: %f, train acc: %f, test acc: %f"%(epoch,train_loss/len(train_data), train_acc/len(train_data),test_acc))
