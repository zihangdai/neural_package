import numpy as np
import scipy.io
import os
from sklearn import utils as skutils

"""Collect common small-scale image datasets

Usage:
    trX, teX, trY, teY = dataset_name(data_dir, preprocess=callable_func)

Notes:
    - The preprocess only applies to data not the label
"""

#### Download link: https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT
def omniglot(data_dir, **kwargs):
    def reshape_data(data):
        return data.reshape((-1, 1, 28, 28), order='fortran')
    omni_raw = scipy.io.loadmat(os.path.join(data_dir, 'chardata.mat'))

    trX = reshape_data(omni_raw['data'].T.astype(float))
    teX = reshape_data(omni_raw['testdata'].T.astype(float))

    trY = omni_raw['targetchar'].flatten()
    teY = omni_raw['testtargetchar'].flatten()

    # There are totally 55 classes
    trY[trY==55] = 0
    teY[teY==55] = 0

    preprocess = kwargs.get('preprocess', False)
    if preprocess and callable(preprocess):
        trX = preprocess(trX)
        teX = preprocess(teX)

    return trX, teX, trY, teY

def omniglot_with_valid_set(data_dir, **kwargs):
    # 1345 data points are used for validation (following IWAE)
    trX, teX, trY, teY = omniglot(data_dir, **kwargs)
    trX, trY = skutils.shuffle(trX, trY,
                               random_state=np.random.RandomState(12345))
    vaX = trX[-1345:]
    vaY = trY[-1345:]
    trX = trX[:-1345]
    trY = trY[:-1345]

    return trX, vaX, teX, trY, vaY, teY

#### Download link: http://yann.lecun.com/exdb/mnist/
def mnist(data_dir, **kwargs):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    preprocess = kwargs.get('preprocess', False)
    if preprocess and callable(preprocess):
        trX = preprocess(trX)
        teX = preprocess(teX)

    return trX, teX, trY, teY 

def mnist_with_valid_set(data_dir, **kwargs):
    trX, teX, trY, teY = mnist(data_dir, **kwargs)

    trX, trY = skutils.shuffle(trX, trY,
                               random_state=np.random.RandomState(12345))
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY

#### Download link: https://www.cs.toronto.edu/~kriz/cifar.html
def cifar10(data_dir, **kwargs):
    # Cifar10 comes with 5 partitions
    def _load_batch_cifar10(data_dir, batch_name):
        """load a batch in the CIFAR-10 format"""
        path = os.path.join(data_dir, batch_name)
        batch = np.load(path)
        data = batch['data']
        labels = batch['labels']
        return data, labels

    # train
    trX, trY = [], []
    for k in xrange(5):
        x, t = _load_batch_cifar10(data_dir, 'data_batch_{}'.format(k + 1)) 
        trX.append(x)
        trY.append(t)

    trX = np.concatenate(trX)
    trY = np.concatenate(trY)

    # test
    teX, teY = _load_batch_cifar10(data_dir, 'test_batch')

    preprocess = kwargs.get('preprocess', False)
    if preprocess and callable(preprocess):
        trX = preprocess(trX)
        teX = preprocess(teX)
    
    return trX, teX, trY, teY

#### Download link: http://ufldl.stanford.edu/housenumbers/
def svhn(data_dir, **kwargs):
    def _load_svhn_split(data_dir, split):
        path = os.path.join(data_dir, '{}_32x32.mat'.format(split))
        data = scipy.io.loadmat(path)
        X = data['X'].transpose(3, 0, 1, 2).astype(float)
        Y = data['y'].reshape((-1))
        Y[Y == 10] = 0
        return X, Y

    trX, trY = _load_svhn_split(data_dir, 'train')
    teX, teY = _load_svhn_split(data_dir, 'test')

    preprocess = kwargs.get('preprocess', False)
    if preprocess and callable(preprocess):
        trX = preprocess(trX)
        teX = preprocess(teX)
    
    return trX, teX, trY, teY

# TODO:
# 3D chairs: https://github.com/mathieuaubry/seeing3Dchairs/
# 3D faces: https://github.com/statismo/statismo