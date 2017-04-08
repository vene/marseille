import os

import dill
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from lightning.classification import SAGAClassifier

from marseille.custom_logging import logging
from marseille.datasets import get_dataset_loader
from marseille.io import load_csr, cache_fname
from marseille.argdoc import DocLabel
from marseille.struct_models import BaseArgumentMixin


class BaselineStruct(BaseArgumentMixin):

    def __init__(self, alpha_link, alpha_prop, l1_ratio, exact_test=False):
        self.alpha_link = alpha_link
        self.alpha_prop = alpha_prop
        self.l1_ratio = l1_ratio
        self.compat_features = False
        self.exact_test = exact_test

    def initialize_labels(self, y_props_flat, y_links_flat):
        self.prop_encoder_ = LabelEncoder().fit(y_props_flat)
        self.link_encoder_ = LabelEncoder().fit(y_links_flat)

        self.n_prop_states = len(self.prop_encoder_.classes_)
        self.n_link_states = len(self.link_encoder_.classes_)

    def fit(self, X_link, y_link, X_prop, y_prop):
        self.initialize_labels(y_prop, y_link)
        y_link = self.link_encoder_.transform(y_link)
        y_prop = self.prop_encoder_.transform(y_prop)

        self.link_clf_ = SAGAClassifier(loss='smooth_hinge', penalty='l1',
                                        tol=1e-4,  max_iter=500,
                                        random_state=0, verbose=0)

        self.prop_clf_ = clone(self.link_clf_)

        alpha_link = self.alpha_link * (1 - self.l1_ratio)
        beta_link = self.alpha_link * self.l1_ratio
        sw = compute_sample_weight('balanced', y_link)
        self.link_clf_.set_params(alpha=alpha_link, beta=beta_link)
        self.link_clf_.fit(X_link, y_link, sample_weight=sw)

        alpha_prop = self.alpha_prop * (1 - self.l1_ratio)
        beta_prop = self.alpha_prop * self.l1_ratio
        self.prop_clf_.set_params(alpha=alpha_prop, beta=beta_prop)
        self.prop_clf_.fit(X_prop, y_prop)
        return self

    def decision_function(self, X_link, X_prop, docs):

        link_offsets = np.cumsum([len(doc.features) for doc in docs])
        y_link_flat = self.link_clf_.decision_function(X_link)

        y_link_marg = np.zeros((len(y_link_flat),
                                len(self.link_encoder_.classes_)))
        link_on, = self.link_encoder_.transform([True])
        y_link_marg[:, link_on] = y_link_flat.ravel()

        Y_link = [y_link_marg[start:end] for start, end
                  in zip(np.append(0, link_offsets), link_offsets)]

        prop_offsets = np.cumsum([len(doc.prop_features) for doc in docs])
        y_prop_marg = self.prop_clf_.decision_function(X_prop)
        Y_prop = [y_prop_marg[start:end] for start, end
                  in zip(np.append(0, prop_offsets), prop_offsets)]

        Y_pred = []
        for y_link, y_prop in zip(Y_link, Y_prop):
            Y_pred.append(DocLabel(y_prop, y_link))

        assert len(Y_pred) == len(docs)

        return Y_pred

    def fast_decode(self, Y_marg, docs, constraints):
        if constraints:
            Y_pred = []
            zero_compat = np.zeros((self.n_prop_states,
                                    self.n_prop_states,
                                    self.n_link_states))

            for doc, y in zip(docs, Y_marg):
                potentials = (y.nodes, y.links, zero_compat, [], [], [])
                y_decoded, _ = self._inference(doc, potentials, relaxed=False,
                                               exact=self.exact_test,
                                               constraints=constraints)
                Y_pred.append(y_decoded)

        else:
            Y_pred = [self._round(y.nodes, y.links, inverse_transform=True)
                      for y in Y_marg]
        return Y_pred

    def predict(self, X_link, X_prop, docs, constraints=""):
        Y_marg = self.decision_function(X_link, X_prop, docs)
        return self.fast_decode(Y_marg, docs, constraints)


def saga_decision_function(dataset, k, link_alpha, prop_alpha, l1_ratio):

    fn = cache_fname("linear_val_df", (dataset, k, link_alpha, prop_alpha,
                                       l1_ratio))

    if os.path.exists(fn):
        logging.info("Loading {}".format(fn))
        with open(fn, "rb") as f:
            return dill.load(f)

    ds = 'erule' if dataset == 'cdcp' else 'ukp-essays'  # sorry
    path = os.path.join("data", "process", ds, "folds", "{}", "{}")

    # sorry again: get val docs
    n_folds = 5 if dataset == 'ukp' else 3
    load, ids = get_dataset_loader(dataset, "train")
    for k_, (_, val) in enumerate(KFold(n_folds).split(ids)):
        if k_ == k:
            break
    val_docs = list(load(ids[val]))

    X_tr_link, y_tr_link = load_csr(path.format(k, 'train.npz'),
                                    return_y=True)
    X_te_link, y_te_link = load_csr(path.format(k, 'val.npz'),
                                    return_y=True)

    X_tr_prop, y_tr_prop = load_csr(path.format(k, 'prop-train.npz'),
                                    return_y=True)
    X_te_prop, y_te_prop = load_csr(path.format(k, 'prop-val.npz'),
                                    return_y=True)

    baseline = BaselineStruct(link_alpha, prop_alpha, l1_ratio)
    baseline.fit(X_tr_link, y_tr_link, X_tr_prop, y_tr_prop)

    Y_marg = baseline.decision_function(X_te_link, X_te_prop, val_docs)

    with open(fn, "wb") as f:
        logging.info("Saving {}".format(fn))
        dill.dump((Y_marg, baseline), f)

    return Y_marg, baseline


def linear_cv_score(dataset, alpha, l1_ratio, constraints):

    fn = cache_fname("linear_cv_score", (dataset, alpha, l1_ratio,
                                         constraints))
    if os.path.exists(fn):
        logging.info("Loading {}".format(fn))
        with open(fn, "rb") as f:
            return dill.load(f)

    load, ids = get_dataset_loader(dataset, split="train")
    n_folds = 5 if dataset == 'ukp' else 3

    scores = []
    for k, (tr, val) in enumerate(KFold(n_folds).split(ids)):
        Y_marg, bl = saga_decision_function(dataset, k, alpha, alpha, l1_ratio)

        val_docs = list(load(ids[val]))
        Y_true = [doc.label for doc in val_docs]
        Y_pred = bl.fast_decode(Y_marg, val_docs, constraints)

        scores.append(bl._score(Y_true, Y_pred))

    with open(fn, "wb") as f:
        logging.info("Saving {}".format(fn))
        dill.dump(scores, f)
    return scores


def main():
    from docopt import docopt

    usage = """

    Usage:
        exp_linear (cdcp|ukp) [--l1-ratio=N --constraints --strict]

    Options:
        --l1-ratio=N    amount of l1 [default: 0]
    """

    args = docopt(usage)

    dataset = 'cdcp' if args['cdcp'] else 'ukp'

    if args['--constraints']:
        constraints = dataset
        if args['--strict']:
            constraints += '+strict'
    else:
        constraints = ""

    l1_ratio = float(args['--l1-ratio'])

    # approximately np.logspace(-5, -1, 9, base=10)
    alphas = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])

    res = [linear_cv_score(dataset, alpha, l1_ratio, constraints)
           for alpha in alphas]

    mean_f1 = np.zeros_like(alphas)
    for ix, scores in enumerate(res):
        mean_scores = np.mean(scores, axis=0)
        _, link_micro, _, prop_micro, _ = mean_scores
        mean_f1[ix] = 0.5 * (link_micro + prop_micro)

    ix = mean_f1.argmax()

    print("Best alpha: ", alphas[ix])
    print("Best scores:")
    print("Link: {:.3f}/{:.3f} Node: {:.3f}/{:.3f} accuracy {:.3f}".format(
        *np.mean(res[ix], axis=0)))


if __name__ == '__main__':
    main()
