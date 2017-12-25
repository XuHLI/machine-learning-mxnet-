from mxnet import ndarray as nd
import mxnet.autograd as ag 
import numpy as np 

# x = np.ones((3,4))

# y = nd.array(x)  # numpy - mexnet
# z = y.asnumpy() # mxnet - numpy

# print([z,y])

# print(type(y))

# x = nd.ones((3,4))
# y = nd.ones((3,4))

# before = id(y)  # id of y

# y = y+x

# print(before)

# print(id(y)==before)

# z = x+y

# nd.elemwise_add(x,y,out=z)
# print(id(z)==before)

#################
## autograd
#################


# x = nd.array([[1,2],[3,4]])
# print(x)

# x.attach_grad()

# with ag.record():
# 	y = x*2
# 	z = y*x

# print(z)

# z.backward()  # gradient of z

# print(x.grad==4*x)


def f(a):
	b =a*2
	while nd.norm(b).asscalar()<1000:
		b=b*2
	if nd.sum(b).asscalar()>0:
		c=b
	else:
		c=100*b
	return c

a = nd.random_normal(shape=3)
a.attach_grad()

with ag.record():
	c =f(a)
c.backward()

print(a.grad)


print(c/a)

