import numpy as np
import os
import theano, theano.tensor as T
import nn
import lasagne

from datautils import cifar10, mnist, svhn, omniglot
from datautils import FixDimIterator, MultiFixDimIterator

def datautil_example():
    def comp_transform(X):
        newX = (X/127.5) - 1.
        return newX.astype(theano.config.floatX)

    def comp_transform_gray(X):
        newX = X/255.
        return newX.astype(theano.config.floatX)

    # data_dir = os.path.join(os.path.expanduser("~"), "Dropbox/Dataset/cifar10/cifar-10-batches-py")
    # trX, teX, trY, teY = cifar10(data_dir, preprocess=comp_transform)

    # data_dir = os.path.join(os.path.expanduser("~"), "Dropbox/Dataset/mnist")
    # trX, teX, trY, teY = mnist(data_dir, preprocess=comp_transform_gray)

    data_dir = os.path.join(os.path.expanduser("~"), "Dropbox/Dataset/svhn")
    trX, teX, trY, teY = svhn(data_dir, preprocess=comp_transform)

    train_iter = FixDimIterator(trX, batch_size=20, shuffle=True)
    for batch in train_iter:
        print batch.shape
        # print np.min(batch), np.max(batch)

    # train_iter = MultiFixDimIterator(trX, trY, batch_size=20, shuffle=True)
    # for batch, label in train_iter:
    #     print batch.shape, label.shape
    #     print np.min(batch), np.max(batch)

def nn_example():
    net = nn.Sequential(
            nn.Linear(784, 256, name='Linear_1'), # manually set a name field if needed
            nn.ReLU(),
            nn.Linear(256, 256, weight=lasagne.init.Normal(1.)),  # similar weight initialization scheme
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax()
        )

    #### The initialization method above is equivalent to the one commented below
    
    # net = nn.Sequential()
    # net.add(nn.Linear(784, 256))
    # net.add(nn.ReLU())
    # net.add(nn.Linear(256, 256))
    # net.add(nn.ReLU())
    # net.add(nn.Linear(256, 10))
    # net.add(nn.Softmax())
    
    print net  # see output: allowing print a mini-strcuture of the network
    
    input = T.matrix()
    prob = net.forward(input)

    params = net.get_parameters()
    for p in params:
        print p  # see output: each parameter is name by "module_name + '_' + param_name"

    predict = theano.function([input], [prob])

    data_dir = os.path.join(os.path.expanduser("~"), "Dropbox/Dataset/mnist")
    trX, teX, trY, teY = mnist(data_dir)

    train_iter = FixDimIterator(trX, batch_size=20, shuffle=True)
    batch = next(train_iter)

    predict(batch)

# nn_example()

def module_attribute_example():
    """
    This example shows a desired feature of Module class:
        - When a Module instance A is assigned as an attribute to another Module class B, 
        class B automatically recognize A as its child.
    """
    class Net(nn.Module):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)

            self.linear = nn.Linear(100, 100, use_bias=False)

            self.conv = nn.Convolutional((2,2), 3, 3)


        def forward(self, input):
            out_linear = self.linear.foward(input)
            out_conv = self.conv.foward(input)

            return out_linear, out_conv

    net = Net()

    # the children modules of net
    print '=' * 120
    for m in net.children():
        print m

    # the children modules of net + itself
    print '=' * 120
    for m in net.modules():
        print m

    # all parameters of net
    print '=' * 120
    params = net.get_parameters()
    for p in params:
        print p

module_attribute_example()