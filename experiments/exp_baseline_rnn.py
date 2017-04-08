import os

import dill
import numpy as np
from sklearn.model_selection import KFold

from marseille.custom_logging import logging
from marseille.datasets import get_dataset_loader, load_embeds
from marseille.io import cache_fname
from marseille.argrnn import BaselineArgumentLSTM


def baseline_argrnn_cv_score(dataset, dynet_weight_decay, mlp_dropout,
                             rnn_dropout, prop_layers, constraints):

    fn = cache_fname("baseline_argrnn_cv_score", (dataset, dynet_weight_decay,
                                                  mlp_dropout, rnn_dropout,
                                                  prop_layers, constraints))
    if os.path.exists(fn):
        logging.info("Cached file already exists.")
        with open(fn, "rb") as f:
            return dill.load(f)

    load, ids = get_dataset_loader(dataset, split="train")
    embeds = load_embeds(dataset)

    scores = []
    Y_pred = []
    score_at_iter = [10, 25, 50, 75, 100]

    n_folds = 5 if dataset == 'ukp' else 3

    for k, (tr, val) in enumerate(KFold(n_folds).split(ids)):
        docs_train = list(load(ids[tr]))
        docs_val = list(load(ids[val]))

        Y_train = [doc.label for doc in docs_train]
        Y_val = [doc.label for doc in docs_val]

        rnn = BaselineArgumentLSTM(lstm_dropout=rnn_dropout,
                                   mlp_dropout=mlp_dropout,
                                   prop_mlp_layers=prop_layers,
                                   max_iter=100,
                                   score_at_iter=score_at_iter,
                                   n_mlp=128,
                                   n_lstm=128,
                                   lstm_layers=2,
                                   link_mlp_layers=1,
                                   embeds=embeds,
                                   link_bilinear=True,
                                   constraints=constraints)

        rnn.fit(docs_train, Y_train, docs_val, Y_val)
        Y_val_pred = rnn.predict(docs_val)
        Y_pred.extend(Y_val_pred)

        scores.append(rnn.scores_)

    with open(fn, "wb") as f:
        dill.dump((scores, score_at_iter, Y_pred), f)

    return scores, score_at_iter, Y_pred


if __name__ == '__main__':
    from docopt import docopt

    usage = """
    Usage:
        exp_baseline_rnn (cdcp|ukp) [\
--dynet-seed N --dynet-weight-decay N --dynet-mem N --prop-layers=N \
--rnn-dropout=N --mlp-dropout=N --constraints --strict]

    Options:
        --dynet-seed=N          random number generator seed for dynet library
        --dynet-weight-decay=N  global weight decay amount for dynet library
        --dynet-mem=N           memory pool size for dynet
        --prop-layers=N         number of prop classifier layers. [default: 2]
        --rnn-dropout=N         dropout ratio in lstm. [default: 0.0]
        --mlp-dropout=N         dropout ratio in mlp. [default: 0.1]
        --constraints           whether to constrain the decoding
        --strict                whether to use strict domain constraints
    """

    args = docopt(usage)

    dataset = 'cdcp' if args['cdcp'] else 'ukp'
    prop_layers = int(args['--prop-layers'])
    rnn_dropout = float(args['--rnn-dropout'])
    mlp_dropout = float(args['--mlp-dropout'])

    if args['--constraints']:
        constraints = dataset
        if args['--strict']:
            constraints += '+strict'
    else:
        constraints = ""

    scores, score_at_iter, _ = baseline_argrnn_cv_score(
        dataset, args['--dynet-weight-decay'], mlp_dropout, rnn_dropout,
        prop_layers, constraints)

    for fold_scores in scores:
        # repeat last result in case we stopped early
        while len(fold_scores) < len(score_at_iter):
            fold_scores.append(fold_scores[-1])

    for iter, score in zip(score_at_iter, np.mean(scores, axis=0)):
        print("iter={} "
              "Link: {:.3f}/{:.3f} "
              "Node: {:.3f}/{:.3f} "
              "accuracy {:.3f}".format(iter, *score),
        )
