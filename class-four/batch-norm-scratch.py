import OpenSSL.SSL
from mxnet import nd
def pure_batch_norm(X,gamma,beta,eps=1e-5):
	assert len(X.shape) in (2,4) # (x,y) or convolution 

	if len(X.shape) == 2:
		mean = X.mean(axis=0)
		# print(mean)
		variance = ((X-mean)**2).mean(axis=0)

	else:
		mean = X.mean(axis=(0,2,3), keepdims=True) # compute mean and variance for each channel
		# print(mean)
		variance = ((X-mean)**2).mean(axis=(0,2,3),keepdims=True)

	# normalization
	X_hat = (X-mean)/nd.sqrt(variance+eps)

	return gamma.reshape(mean.shape)*X_hat+beta.reshape(mean.shape)

A = nd.arange(6).reshape((3,2))
print(A)

print(pure_batch_norm(A,gamma=nd.array([1,1]), beta=nd.array([0,0])))

B = nd.arange(36).reshape((2,2,3,3))
print(B)

print(pure_batch_norm(B,gamma=nd.array([1,1]), beta=nd.array([0,0])))

def batch_norm(X,gamma,beta, is_training,moving_mean, moving_variance, eps =1e-5, moving_momentum = 0.9):
	assert len(X.shape) in (2,4)

    # batch_size x feature
	if len(X.shape) == 2:
		mean = X.mean(axis=0)
		# print(mean)
		variance = ((X-mean)**2).mean(axis=0)
    # 2D convolution: batch_size x channels x height x weight
	else:
		mean = X.mean(axis=(0,2,3), keepdims=True) # compute mean and variance for each channel
		# print(mean)
		variance = ((X-mean)**2).mean(axis=(0,2,3),keepdims=True)

		# for correct 
		moving_mean = moving_mean.reshape(mean.shape)
		moving_variance = moving_variance.reshape(mean.shape)

	# normalization

	if is_training:
		X_hat = (X-mean)/nd.sqrt(variance+eps)
		# update global mean and variance
		moving_mean[:] = moving_momentum * moving_mean + (
			1.0-moving_momentum) * mean
		moving_variance[:] = moving_momentum * moving_variance + (
			1.0-moving_momentum) * variance
	else:
		X_hat = (X-moving_mean)/nd.sqrt(moving_variance+eps)


	return gamma.reshape(mean.shape)*X_hat+beta.reshape(mean.shape)
