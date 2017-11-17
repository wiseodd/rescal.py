import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Example Kinships')

import numpy as np
from numpy import dot, array, zeros, setdiff1d
from numpy.linalg import norm
from numpy.random import shuffle
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve, auc
from rescal import rescal_als
import argparse


parser = argparse.ArgumentParser(
    description='RESCAL reconstrution experiment'
)

parser.add_argument('--rank', type=int, default='20', metavar='',
                    help='rank of reconstructed tensor (default: 20)')

args = parser.parse_args()


def unfold(X, k=0):
    """ Unfold tensor X w.r.t mode k (default: 0)"""
    return X.swapaxes(0, k).reshape(X.shape[k], -1)


# load data
mat = loadmat('data/alyawarradata.mat')
K = array(mat['Rs'], np.float32)  # Original tensor
e, k = K.shape[0], K.shape[2]

# construct array for rescal
T = [lil_matrix(K[:, :, i]) for i in range(k)]

print('\nTensor size: %d x %d x %d' % (
    T[0].shape + (len(T),))
)

print('\nTraining RESCAL')
print('===============')

E, R, _, _, _ = rescal_als(
    T, rank=args.rank, init='nvecs', conv=1e-3,
    lambda_A=10, lambda_R=10
)

print('===============\n')

print('=====================')
print('Original tensor stats')
print('=====================')

nuc_norm = np.sum([np.linalg.norm(unfold(K, k), 'nuc') for k in range(len(K.shape))])
print('Sum of Nuclear Norm: {:.4f}'.format(nuc_norm))

l1_norm = np.linalg.norm(K.ravel(), 1)
print('L1 Norm (vectorized): {:.4f}'.format(l1_norm))

frob_norm = np.linalg.norm(K)  # Frobenius by default
print('Frobenius Norm: {:.4f}'.format(frob_norm))


print('\n==========================')
print('Reconstructed tensor stats')
print('==========================')

E, W = np.array(E), np.array(R)

# Reconstruction:
n_e = e
n_r = k

X = np.zeros([n_e, n_e, n_r])  # Reconstructed tensor

for i in range(n_r):
    X[:, :, i] = E @ W[i] @ E.T

nuc_norm = np.sum([np.linalg.norm(unfold(X, k), 'nuc') for k in range(len(X.shape))])
print('Sum of Nuclear Norm: {:.4f}'.format(nuc_norm))

l1_norm = np.linalg.norm(X.ravel(), 1)
print('L1 Norm (vectorized): {:.4f}'.format(l1_norm))

frob_norm = np.linalg.norm(X)  # Frobenius by default
print('Frobenius Norm: {:.4f}'.format(frob_norm))


print('\n======================')
print('Residual tensor stats')
print('======================')

X_res = K - X

nuc_norm = np.sum([np.linalg.norm(unfold(X_res, k), 'nuc') for k in range(len(X_res.shape))])
print('Sum of Nuclear Norm: {:.4f}'.format(nuc_norm))

l1_norm = np.linalg.norm(X_res.ravel(), 1)
print('L1 Norm (vectorized): {:.4f}'.format(l1_norm))

frob_norm = np.linalg.norm(X_res)  # Frobenius by default
print('Frobenius Norm: {:.4f}'.format(frob_norm))