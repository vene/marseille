# Link prediction baselines.

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from marseille.io import load_csr
from marseille.custom_logging import logging


def main():
    from docopt import docopt

    usage = """
    Usage:
        baselines (cdcp|ukp) [--n-folds=N]

    Options:
        --n-folds=N  number of cross-val folds to generate. [default: 3]
    """

    args = docopt(usage)
    n_folds = int(args['--n-folds'])

    all_true = []
    all_false = []
    adjacent = []
    adjacent_ltr = []
    adjacent_rtl = []

    if args['cdcp']:
        path = os.path.join("data", "process", "erule", "folds", "{}", "{}")
    elif args['ukp']:
        path = os.path.join("data", "process", "ukp-essays", "folds", "{}",
                            "{}")

    for k in range(n_folds):
        fname = path.format(k, 'val.npz')
        logging.info("Loading sparse vectorized file {}".format(fname))
        X_te, y_te = load_csr(fname, return_y=True)

        with open(path.format(k, "fnames.txt")) as f:
            fnames = [line.strip() for line in f]

        props_between = fnames.index('nrm__props_between')
        src_precedes_trg = fnames.index('raw__src_precedes_trg')
        trg_precedes_src = fnames.index('raw__trg_precedes_src')

        y_all_true = np.ones_like(y_te)
        y_all_false = np.zeros_like(y_te)

        y_adj = ~(X_te[:, props_between] != 0).A.ravel()
        is_src_first = X_te[:, src_precedes_trg].astype(np.bool).A.ravel()
        is_trg_first = X_te[:, trg_precedes_src].astype(np.bool).A.ravel()

        y_adj_ltr = y_adj & is_src_first
        y_adj_rtl = y_adj & is_trg_first

        def _score(y):
            p, r, f, _ = precision_recall_fscore_support(y_te, y, pos_label=1,
                                                         average='binary')
            return p, r, f

        all_true.append(_score(y_all_true))
        all_false.append(_score(y_all_false))
        adjacent.append(_score(y_adj))
        adjacent_ltr.append(_score(y_adj_ltr))
        adjacent_rtl.append(_score(y_adj_rtl))

    preds = (all_false, all_true, adjacent, adjacent_ltr, adjacent_rtl)
    preds = [np.array(x).mean(axis=0) for x in preds]
    names = ["All false", "All true", "Adjacent", "Adj s -> t", "Adj t <- s"]

    for name, scores in zip(names, preds):
        print("{:18} {:.4f} {:.4f} {:.4f}".format(name, *scores))


if __name__ == '__main__':
    main()
