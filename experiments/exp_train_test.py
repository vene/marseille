"""Final experiments, trained on the full training set.

This uses hyperparameters chosen by cross-validation from exp_* """

import os
import dill
import pickle

import numpy as np

from marseille.datasets import get_dataset_loader, load_embeds
from marseille.custom_logging import logging
from marseille.argrnn import BaselineArgumentLSTM, ArgumentLSTM
from marseille.io import load_csr

from .exp_svmstruct import fit_predict as fit_pred_pystruct
from .exp_linear import BaselineStruct


hyperparams = {
    'linear': {
        'bare': {
            'cdcp': {'alpha': 0.001},
            'ukp': {'alpha': 0.01}
        },
        'full':  {
            'cdcp': {'alpha': 0.001},
            'ukp': {'alpha': 0.01}
        },
        'strict':  {
            'cdcp': {'alpha': 0.001},
            'ukp': {'alpha': 0.01}
        }
    },
    'linear-struct': {
        'bare': {
            'cdcp': {'C': 0.3},
            'ukp': {'C': 0.01}
        },
        'full':  {
            'cdcp': {'C': 0.1},
            'ukp': {'C': 0.01}
        },
        'strict':  {
            'cdcp': {'C': 0.1},
            'ukp': {'C': 0.03}
        }
    },
    'rnn': {
        'bare': {
            'cdcp': {'max_iter': 25, 'mlp_dropout': 0.15},
            'ukp': {'max_iter': 100, 'mlp_dropout': 0.05}
        },
        'full':  {
            'cdcp': {'max_iter': 25, 'mlp_dropout': 0.05},
            'ukp': {'max_iter': 50, 'mlp_dropout': 0.15}
        },
        'strict':  {
            'cdcp': {'max_iter': 25, 'mlp_dropout': 0.05},
            'ukp': {'max_iter': 25, 'mlp_dropout': 0.1}
        }
    },
    'rnn-struct': {
        'bare': {
            'cdcp': {'max_iter': 25, 'mlp_dropout': 0.25},
            'ukp': {'max_iter': 100, 'mlp_dropout': 0.15}
        },
        'full':  {
            'cdcp': {'max_iter': 100, 'mlp_dropout': 0.25},
            'ukp': {'max_iter': 10, 'mlp_dropout': 0.05}
        },
        'strict':  {
            'cdcp': {'max_iter': 10, 'mlp_dropout': 0.2},
            'ukp': {'max_iter': 10, 'mlp_dropout': 0.15}
        }
    }
}

exact_test = True

if __name__ == '__main__':
    from docopt import docopt

    usage = """
    Usage:
        exp_train_test (cdcp|ukp) --method=M --model=N [--dynet-seed N --dynet-mem N]

    Options:
        --method: one of (linear, linear-struct, rnn, rnn-struct)
        --model: one of (bare, full, strict)
    """

    args = docopt(usage)

    dataset = 'cdcp' if args['cdcp'] else 'ukp'
    method = args['--method']
    model = args['--model']

    params = hyperparams[method][model][dataset]

    load_tr, ids_tr = get_dataset_loader(dataset, split="train")
    load_te, ids_te = get_dataset_loader(dataset, split="test")

    train_docs = list(load_tr(ids_tr))
    test_docs = list(load_te(ids_te))

    logging.info("{} {} on {} ({})".format(method, model, dataset, params))

    filename = os.path.join('test_results',
                            'exact={}_{}_{}_{}'.format(exact_test,
                                                       dataset,
                                                       method,
                                                       model))
    if not os.path.exists('test_results'):
        os.makedirs('test_results')

    # logic for constraints and compat features
    # note that compat_features and second_order aren't used
    # if the model isn't structured, but it's more readable this way.

    if model == 'bare':
        constraints = ''
        compat_features = False
        second_order = False
    elif model == 'full':
        constraints = dataset
        compat_features = True
        second_order = True
    elif model == 'strict':
        constraints = '{}+strict'.format(dataset)
        compat_features = True
        second_order = True
    else:
        raise ValueError('Invalid model: {}'.format(model))

    # logic for which second order features to use, if any
    grandparents = second_order and dataset == 'ukp'
    coparents = second_order
    siblings = second_order and dataset == 'cdcp'

    if method == 'linear':
        ds = 'erule' if dataset == 'cdcp' else 'ukp-essays'
        path = os.path.join("data", "process", ds, "folds", "traintest", "{}")
        X_tr_link, y_tr_link = load_csr(path.format('train.npz'),
                                        return_y=True)
        X_te_link, y_te_link = load_csr(path.format('test.npz'),
                                        return_y=True)

        X_tr_prop, y_tr_prop = load_csr(path.format('prop-train.npz'),
                                        return_y=True)
        X_te_prop, y_te_prop = load_csr(path.format('prop-test.npz'),
                                        return_y=True)

        baseline = BaselineStruct(alpha_link=params['alpha'],
                                  alpha_prop=params['alpha'],
                                  l1_ratio=0,
                                  exact_test=exact_test)
        baseline.fit(X_tr_link, y_tr_link, X_tr_prop, y_tr_prop)
        Y_pred = baseline.predict(X_te_link, X_te_prop, test_docs, constraints)

        with open('{}.model.pickle'.format(filename), "wb") as fp:
            pickle.dump(baseline, fp)

        np.save('{}.model'.format(filename),
                (baseline.prop_clf_.coef_, baseline.link_clf_.coef_))

    elif method == 'linear-struct':

        clf, Y_te, Y_pred, vects = fit_pred_pystruct(train_docs, test_docs,
            dataset=dataset, class_weight='balanced',
            constraints=constraints, compat_features=compat_features,
            second_order=second_order, coparents=coparents,
            grandparents=grandparents, siblings=siblings,
            exact_test=exact_test, return_vectorizers=True, **params)

        with open('{}.vectorizers.pickle'.format(filename), "wb") as fp:
            pickle.dump(vects, fp)

        with open('{}.model.pickle'.format(filename), "wb") as fp:
            pickle.dump(clf, fp)

        np.save('{}.model'.format(filename), clf.w)

    elif method == "rnn":

        Y_train = [doc.label for doc in train_docs]
        Y_te = [doc.label for doc in test_docs]

        embeds = load_embeds(dataset)

        rnn = BaselineArgumentLSTM(lstm_dropout=0,
                                   prop_mlp_layers=2,
                                   score_at_iter=None,
                                   n_mlp=128,
                                   n_lstm=128,
                                   lstm_layers=2,
                                   link_mlp_layers=1,
                                   embeds=embeds,
                                   link_bilinear=True,
                                   constraints=constraints,
                                   exact_test=exact_test,
                                   **params)

        rnn.fit(train_docs, Y_train)
        with open('{}.model.pickle'.format(filename), "wb") as fp:
            pickle.dump(rnn, fp)
        rnn.save('{}.model.dynet'.format(filename))
        Y_pred = rnn.predict(test_docs)

    elif method == "rnn-struct":
        Y_train = [doc.label for doc in train_docs]
        Y_te = [doc.label for doc in test_docs]

        embeds = load_embeds(dataset)

        rnn = ArgumentLSTM(lstm_dropout=0,
                           prop_mlp_layers=2,
                           score_at_iter=None,
                           n_mlp=128,
                           n_lstm=128,
                           lstm_layers=2,
                           link_mlp_layers=1,
                           embeds=embeds,
                           link_bilinear=True,
                           class_weight='balanced',
                           second_order_multilinear=True,
                           exact_inference=False,
                           constraints=constraints,
                           compat_features=compat_features,
                           grandparent_layers=(1 if grandparents else 0),
                           coparent_layers=(1 if coparents else 0),
                           sibling_layers=(1 if siblings else 0),
                           exact_test=exact_test,
                           **params)

        rnn.fit(train_docs, Y_train)
        with open('{}.model.pickle'.format(filename), "wb") as fp:
            pickle.dump(rnn, fp)
        rnn.save('{}.model.dynet'.format(filename))
        Y_pred = rnn.predict(test_docs)

    with open('{}.predictions.dill'.format(filename), "wb") as f:
        dill.dump(Y_pred, f)
