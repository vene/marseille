import os

import dill
import numpy as np
from sklearn.model_selection import KFold
from pystruct.learners import FrankWolfeSSVM

from marseille.argdoc import DocStructure
from marseille.custom_logging import logging
from marseille.datasets import get_dataset_loader
from marseille.io import cache_fname
from marseille.struct_models import ArgumentGraphCRF
from marseille.vectorize import (add_pmi_features, stats_train, prop_vectorizer,
                                 link_vectorizer, second_order_vectorizer)


def _vectorize(doc, pmi_incoming, pmi_outgoing, prop_vect, link_vect,
               second_order_vect=None):
    for f in doc.features:
        add_pmi_features(f, pmi_incoming, pmi_outgoing)

    X_prop = prop_vect.transform(doc.prop_features)
    X_link = link_vect.transform(doc.features)

    if second_order_vect is not None:
        n_sec_ord_features = len(second_order_vect.get_feature_names())
        if doc.second_order_features:
            X_sec_ord = second_order_vect.transform(doc.second_order_features)
        else:
            X_sec_ord = np.empty((0, n_sec_ord_features))
        return DocStructure(doc, X_prop, X_link, X_sec_ord)

    return DocStructure(doc, X_prop, X_link)


def fit_predict(train_docs, test_docs, dataset, C, class_weight, constraints,
                compat_features, second_order, coparents, grandparents,
                siblings, exact_test=False):
    stats = stats_train(train_docs)
    prop_vect, _ = prop_vectorizer(train_docs,
                                   which=dataset,
                                   stats=stats,
                                   n_most_common_tok=None,
                                   n_most_common_dep=2000,
                                   return_transf=True)
    link_vect = link_vectorizer(train_docs, stats, n_most_common=500)

    sec_ord_vect = (second_order_vectorizer(train_docs)
                    if second_order else None)

    _, _, _, pmi_in, pmi_out = stats

    def _transform_x_y(docs):
        X = [_vectorize(doc, pmi_in, pmi_out, prop_vect, link_vect,
                        sec_ord_vect)
             for doc in docs]
        Y = [doc.label for doc in docs]
        return X, Y

    X_tr, Y_tr = _transform_x_y(train_docs)
    X_te, Y_te = _transform_x_y(test_docs)

    model = ArgumentGraphCRF(class_weight=class_weight,
                             constraints=constraints,
                             compat_features=compat_features,
                             coparents=coparents,
                             grandparents=grandparents,
                             siblings=siblings)

    clf = FrankWolfeSSVM(model, C=C, random_state=0, verbose=1,
                         check_dual_every=25,
                         show_loss_every=25,
                         max_iter=100,
                         tol=0)

    clf.fit(X_tr, Y_tr)

    if exact_test:
        clf.model.exact = True
    Y_pred = clf.predict(X_te)

    return clf, Y_te, Y_pred


def svmstruct_cv_score(dataset, C, class_weight, constraints,
                       compat_features, second_order_features):

    fn = cache_fname("svmstruct_cv_score", (dataset, C, class_weight,
                                            constraints, compat_features,
                                            second_order_features))

    if os.path.exists(fn):
        logging.info("Cached file already exists.")
        with open(fn, "rb") as f:
            return dill.load(f)

    load, ids = get_dataset_loader(dataset, split="train")

    n_folds = 5 if dataset == 'ukp' else 3

    # below are boolean logical ops
    grandparents = second_order_features and dataset == 'ukp'
    coparents = second_order_features
    siblings = second_order_features and dataset == 'cdcp'

    scores = []
    all_Y_pred = []

    for k, (tr, val) in enumerate(KFold(n_folds).split(ids)):
        train_docs = list(load(ids[tr]))
        val_docs = list(load(ids[val]))

        clf, Y_val, Y_pred = fit_predict(train_docs, val_docs, dataset, C,
                                         class_weight,
                                         constraints, compat_features,
                                         second_order_features, grandparents,
                                         coparents, siblings)
        all_Y_pred.extend(Y_pred)
        scores.append(clf.model._score(Y_val, Y_pred))

    with open(fn, "wb") as f:
        dill.dump((scores, all_Y_pred), f)

    return scores, all_Y_pred


def main():
    from docopt import docopt

    usage = """
    Usage:
        exp_svmstruct (cdcp|ukp) --C=N [--balanced --constraints --strict
        --compat-features --second-order-features]
    """

    args = docopt(usage)
    C = float(args['--C'])
    dataset = 'cdcp' if args['cdcp'] else 'ukp'
    cw = 'balanced' if args['--balanced'] else None

    if args['--constraints']:
        constraints = dataset
        if args['--strict']:
            constraints += '+strict'
    else:
        constraints = ""

    scores, _ = svmstruct_cv_score(dataset, C, cw, constraints,
                                   args['--compat-features'],
                                   args['--second-order-features'])

    link_macro, link_micro, node_macro, node_micro, acc = np.mean(scores,
                                                                  axis=0)
    print("Link: {:.3f}/{:.3f} Node: {:.3f}/{:.3f} accuracy {:.3f}".format(
        link_macro, link_micro, node_macro, node_micro, acc))


if __name__ == '__main__':
    main()
