from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from module import Sequential, Linear, SoftMax, ClassNLLCriterion, sgd_momentum, get_batches, net_image


N = 500

X1 = 4*np.random.randn(N,2) + np.array([2,2])
X2 = np.random.randn(N,2) + np.array([-2,-2])

Y = np.concatenate([np.ones(N),np.zeros(N)])[:,None]
Y = np.hstack([Y, 1-Y])

X = np.vstack([X1,X2])
plt.scatter(X[:,0],X[:,1], c = Y[:,0], edgecolors= 'none')
plt.show()

net = Sequential()
net.add(Linear(2, 8))
net.add(SoftMax())
net.add(Linear(8, 2))
net.add(SoftMax())

criterion = ClassNLLCriterion()

print(net)

optimizer_config = {'learning_rate' : 1e-1, 'momentum': 0.9}
optimizer_state = {}

# Looping params
n_epoch = 20
batch_size = 128


# batch generator


loss_history = []
#for i in range(n_epoch):
#    for x_batch, y_batch in get_batches((X, Y), batch_size):
for i in range(6):
    for x_batch, y_batch in get_batches((X[:100,:100], Y[:100]), 20):
        net.zeroGradParameters()

        # Forward
        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)
        # print (predictions, y_batch)
        # Backward
        dp = criterion.backward(predictions, y_batch)

        net.backward(x_batch, dp)

        # print ('as', net.getParameters(), net.getGradParameters())
        # Update weights
        sgd_momentum(net.getParameters(),
                     net.getGradParameters(),
                     optimizer_config,
                     optimizer_state)
        # print ('as2', net.getParameters(), net.getGradParameters())

        loss_history.append(loss)

    # Visualize
    display.clear_output(wait=True)
    #plt.figure(figsize=(8, 6))

    #plt.title("Training loss")
    #plt.xlabel("#iteration")
    #plt.ylabel("loss")
    #plt.plot(loss_history, 'b')
    print('Current loss: %f' % loss)
    plt.imshow(net_image(net=net))
    plt.show()

    print('Current loss: %f' % loss)