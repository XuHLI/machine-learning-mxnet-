from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20,kernel_size=5,activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2,strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(128,activation='relu'))
    net.add(gluon.nn.Dense(10))


import sys
sys.path.append('..')
import utils

ctx = utils.try_gpu()
net.initialize(ctx=ctx)

print('initialize weight on', ctx)


from mxnet import autograd as autograd
from mxnet import nd

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.6})

for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)  # make sure that it is on gpu
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)

        train_loss +=nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output,label)
    test_acc = utils.evaluate_accuracy(test_data,net,ctx)

    print("Epoch %d. Loss: %f, Train acc %f, Test acc: %f"%(
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc
    ))