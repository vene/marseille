# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

from hashlib import sha1
import dill
import numpy as np
from scipy.sparse import csr_matrix

from marseille.custom_logging import logging

def save_csr(fname, X, y=None):
    X = X.tocsr()
    X.sort_indices()
    X.eliminate_zeros()
    data = dict(data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)

    if y is not None:
        data['y'] = y

    np.savez(fname, **data)


def load_csr(f, return_y=False):
    npz = np.load(f)
    X = csr_matrix((npz['data'], npz['indices'], npz['indptr']),
                   shape=npz['shape'])

    if return_y:
        return X, npz['y']
    else:
        return X


def cache_fname(key, args):
    arghash = sha1(repr(args).encode('utf-8')).hexdigest()
    return "res/{}_{}.dill".format(key, arghash)


def load_results(key, args):
    fn = cache_fname(key, args)
    with open(fn, "rb") as f:
        return dill.load(f)
