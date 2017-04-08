# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

import numpy as np
import dynet as dy

from nose.tools import assert_almost_equal

from marseille.dynet_utils import MultilinearFactored


def test_multilinear_forward():
    model = dy.Model()

    a, b, c = np.random.RandomState(0).randn(3, 100)
    ml = MultilinearFactored(n_features=100, n_inputs=3, n_components=5,
                             model=model)
    dy_fwd = ml(dy.inputVector(a),
                dy.inputVector(b),
                dy.inputVector(c)).value()

    U = [dy.parameter(u).value() for u in ml.get_components()]
    expected = np.dot(U[0], a)
    expected *= np.dot(U[1], b)
    expected *= np.dot(U[2], c)
    expected = np.sum(expected)

    assert_almost_equal(expected, dy_fwd, 4)
