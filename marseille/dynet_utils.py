# Author: Vlad Niculae <vlad@vene.ro>
# Author: Tianze Shi
# License: BSD 3-clause

import dynet as dy
import numpy as np


class Dense(dy.Saveable):
    def __init__(self, activation, shape, model):
        self.activation = activation
        self.W = model.add_parameters(shape)
        self.b = model.add_parameters(shape[0])
        self.shape = shape

    def __call__(self, x):
        b = dy.parameter(self.b)
        W = dy.parameter(self.W)
        out = b + W * x
        if self.activation:
            out = self.activation(out)
        return out

    def __repr__(self):
        return "Dense(activation={}, shape={}))".format(self.activation,
                                                        self.shape)

    def get_components(self):
        return [self.W, self.b]

    def restore_components(self, components):
        self.W, self.b = components


class MultiLayerPerceptron(dy.Saveable):
    def __init__(self, dims, activation, model):
        self.layers = []
        self.dropout = 0.0
        last = len(dims) - 2
        for k, (n_in, n_out) in enumerate(zip(dims, dims[1:]), 0):
            self.layers.append(Dense(activation if k < last else None,
                                     shape=(n_out, n_in),
                                     model=model))

    def __call__(self, x):
        for k, layer in enumerate(self.layers, 1):
            x = layer(x)
            if self.dropout > 0 and k != len(self.layers):
                x = dy.dropout(x, self.dropout)
        return x

    def get_components(self):
        return self.layers

    def restore_components(self, components):
        self.layers = components


class Bilinear(dy.Saveable):
    def __init__(self, n_in, n_out, model):
        self.W = model.parameters_from_numpy(np.eye(n_in))
        self.w_x = model.add_parameters(n_in)
        self.w_y = model.add_parameters(n_in)
        self.b = model.add_parameters(n_out)
        self.n_out = n_out

    def __call__(self, x, y):
        W = dy.parameter(self.W)
        w_x = dy.parameter(self.w_x)
        w_y = dy.parameter(self.w_y)
        b = dy.parameter(self.b)

        out = dy.transpose(x) * W * y
        out += dy.dot_product(w_x, x)
        out += dy.dot_product(w_y, y)
        out = dy.concatenate([dy.scalarInput(0)] * (self.n_out - 1) + [out])
        out += b

        return out

    def get_components(self):
        return [self.W, self.w_x, self.w_y, self.b]

    def restore_components(self, components):
        [self.W, self.w_x, self.w_y, self.b] = components


class MultilinearFactored(dy.Saveable):
    def __init__(self, n_features, n_inputs, n_components, model):
        self.U = [model.add_parameters((n_components, n_features))
                  for _ in range(n_inputs)]

    def __call__(self, *args):
        U = [dy.parameter(U_) for U_ in self.U]

        out = U[0] * args[0]
        for x, u in zip(args[1:], U[1:]):
            out = dy.cmult(out, u * x)

        out = dy.sum_cols(dy.transpose(out))
        return out

    def get_components(self):
        return self.U

    def restore_components(self, components):
        self.U = components
