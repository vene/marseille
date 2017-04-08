import os

import dill
import numpy as np
from sklearn.model_selection import KFold

from marseille.custom_logging import logging
from marseille.datasets import get_dataset_loader, load_embeds
from marseille.io import cache_fname
from marseille.argrnn import ArgumentLSTM


def argrnn_cv_score(dataset, dynet_weight_decay, mlp_dropout,
                    rnn_dropout, prop_layers, class_weight, constraints,
                    compat_features, second_order):

    fn = cache_fname("argrnn_cv_score", (dataset, dynet_weight_decay,
                                         mlp_dropout, rnn_dropout, prop_layers,
                                         class_weight, constraints,
                                         compat_features, second_order))

    if os.path.exists(fn):
        logging.info("Cached file already exists.")
        with open(fn, "rb") as f:
            return dill.load(f)

    load, ids = get_dataset_loader(dataset, split="train")
    embeds = load_embeds(dataset)

    grandparent_layers = 1 if second_order and dataset == 'ukp' else 0
    coparent_layers = 1 if second_order else 0
    sibling_layers = 1 if second_order and dataset == 'cdcp' else 0

    scores = []
    all_Y_pred = []
    score_at_iter = [10, 25, 50, 75, 100]

    n_folds = 5 if dataset == 'ukp' else 3

    for k, (tr, val) in enumerate(KFold(n_folds).split(ids)):
        docs_train = list(load(ids[tr]))
        docs_val = list(load(ids[val]))

        Y_train = [doc.label for doc in docs_train]
        Y_val = [doc.label for doc in docs_val]

        rnn = ArgumentLSTM(lstm_dropout=rnn_dropout,
                           mlp_dropout=mlp_dropout,
                           compat_features=compat_features,
                           constraints=constraints,
                           prop_mlp_layers=prop_layers,
                           coparent_layers=coparent_layers,
                           grandparent_layers=grandparent_layers,
                           sibling_layers=sibling_layers,
                           class_weight=class_weight,
                           second_order_multilinear=True,
                           max_iter=100,
                           score_at_iter=score_at_iter,
                           n_mlp=128,
                           n_lstm=128,
                           lstm_layers=2,
                           link_mlp_layers=1,
                           embeds=embeds,
                           exact_inference=False,
                           link_bilinear=True)

        rnn.fit(docs_train, Y_train, docs_val, Y_val)
        Y_val_pred = rnn.predict(docs_val)
        all_Y_pred.extend(Y_val_pred)
        scores.append(rnn.scores_)

    with open(fn, "wb") as f:
        dill.dump((scores, score_at_iter, all_Y_pred), f)

    return scores, score_at_iter, all_Y_pred


if __name__ == '__main__':
    from docopt import docopt

    usage = """
    Usage:
        exp_rnn (cdcp|ukp) [\
--dynet-seed N --dynet-weight-decay N --dynet-mem N --prop-layers=N \
--rnn-dropout=N --mlp-dropout=N --balanced --constraints --strict \
--compat-features --second-order]

    Options:
        --dynet-seed=N          random number generator seed for dynet library
        --dynet-weight-decay=N  global weight decay amount for dynet library
        --dynet-mem=N           memory pool size for dynet
        --prop-layers=N         number of prop classifier layers. [default: 2]
        --rnn-dropout=N         dropout ratio in lstm. [default: 0.0]
        --mlp-dropout=N         dropout ratio in mlp. [default: 0.1]
        --balanced              whether to reweight class costs by freq
        --constraints           whether to constrain the decoding
        --strict                whether to use strict domain constraints
        --compat-features       whether to use features for compat factors
        --second-order          whether to use coparent / grandpa / siblings
    """

    args = docopt(usage)

    dataset = 'cdcp' if args['cdcp'] else 'ukp'
    prop_layers = int(args['--prop-layers'])
    rnn_dropout = float(args['--rnn-dropout'])
    mlp_dropout = float(args['--mlp-dropout'])
    cw = 'balanced' if args['--balanced'] else None

    if args['--constraints']:
        constraints = dataset
        if args['--strict']:
            constraints += '+strict'
    else:
        constraints = ""

    scores, score_at_iter, _ = argrnn_cv_score(dataset,
                                               args['--dynet-weight-decay'],
                                               mlp_dropout,
                                               rnn_dropout,
                                               prop_layers,
                                               cw,
                                               constraints,
                                               args['--compat-features'],
                                               args['--second-order'])

    for iter, score in zip(score_at_iter, np.mean(scores, axis=0)):
        print("iter={} "
              "Link: {:.3f}/{:.3f} "
              "Node: {:.3f}/{:.3f} "
              "accuracy {:.3f}".format(iter, *score),
        )
