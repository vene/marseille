"""Baseline for comparability: linear svm by SAGA. Uses structured
format."""

import os
from hashlib import sha1
import warnings
from collections import Counter

import dill
import numpy as np

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
from lightning.classification import SAGAClassifier

from marseille.custom_logging import logging
from marseille.io import load_csr
from marseille.argdoc import DocLabel, CdcpArgumentationDoc, UkpEssayArgumentationDoc
from marseille.struct_models import BaseArgumentMixin
from marseille.datasets import ukp_train_ids, cdcp_train_ids


def saga_cv(which, alphas, l1_ratio):

    if which == 'cdcp':
        n_folds = 3
        path = os.path.join("data", "process", "erule", "folds", "{}", "{}")
    elif which == 'ukp':
        n_folds = 5
        path = os.path.join("data", "process", "ukp-essays", "folds", "{}",
                            "{}")
    else:
        raise ValueError

    clf_link = SAGAClassifier(loss='smooth_hinge', penalty='l1', tol=1e-4,
                              max_iter=100, random_state=0, verbose=0)
    clf_prop = clone(clf_link)

    link_scores = np.zeros((n_folds, len(alphas)))
    prop_scores = np.zeros_like(link_scores)

    for k in range(n_folds):
        X_tr_link, y_tr_link = load_csr(path.format(k, 'train.npz'),
                                        return_y=True)
        X_te_link, y_te_link = load_csr(path.format(k, 'val.npz'),
                                        return_y=True)

        X_tr_prop, y_tr_prop = load_csr(path.format(k, 'prop-train.npz'),
                                        return_y=True)
        X_te_prop, y_te_prop = load_csr(path.format(k, 'prop-val.npz'),
                                        return_y=True)

        le = LabelEncoder()
        y_tr_prop_enc = le.fit_transform(y_tr_prop)
        y_te_prop_enc = le.transform(y_te_prop)

        link_sw = compute_sample_weight('balanced', y_tr_link)

        for j, alpha in enumerate(alphas):

            beta = alpha * l1_ratio
            alpha *= 1 - l1_ratio
            clf_link.set_params(alpha=alpha, beta=beta)
            clf_prop.set_params(alpha=alpha, beta=beta)

            clf_link.fit(X_tr_link, y_tr_link, sample_weight=link_sw)
            y_pred_link = clf_link.predict(X_te_link)

            clf_prop.fit(X_tr_prop, y_tr_prop_enc)
            y_pred_prop = clf_prop.predict(X_te_prop)

            with warnings.catch_warnings() as w:
                warnings.simplefilter('ignore')
                link_f = f1_score(y_te_link, y_pred_link, average='binary')
                prop_f = f1_score(y_te_prop_enc, y_pred_prop, average='macro')

            link_scores[k, j] = link_f
            prop_scores[k, j] = prop_f

    return link_scores, prop_scores


def saga_cv_cache(*args):

    arghash = sha1(repr(args).encode('utf-8')).hexdigest()
    fn = "res/baseline_linear_{}.dill".format(arghash)

    try:
        with open(fn, 'rb') as f:
            out = dill.load(f)
        logging.info("Loaded cached version.")
    except FileNotFoundError:
        logging.info("Computing...")
        out = saga_cv(*args)
        with open(fn, 'wb') as f:
            dill.dump(out, f)

    return out


class BaselineStruct(BaseArgumentMixin):

    def __init__(self, alpha_link, alpha_prop, l1_ratio):
        self.alpha_link = alpha_link
        self.alpha_prop = alpha_prop
        self.l1_ratio = l1_ratio
        self.compat_features = False

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


def saga_score_struct(which, link_alpha, prop_alpha, l1_ratio,
                      decode=False):

    if which == 'cdcp':
        n_folds = 3
        ids = np.array(cdcp_train_ids)
        path = os.path.join("data", "process", "erule", "folds", "{}", "{}")
        _tpl = os.path.join("data", "process", "erule", "{}", "{:05d}")
        _load = lambda which, ks: (CdcpArgumentationDoc(_tpl.format(which, k))
                                   for k in ks)
    elif which == 'ukp':
        n_folds = 5
        ids = np.array(ukp_train_ids)
        path = os.path.join("data", "process", "ukp-essays", "folds", "{}",
                            "{}")
        _tpl = os.path.join("data", "process", "ukp-essays", "essay{:03d}")
        _load = lambda which, ks: (UkpEssayArgumentationDoc(_tpl.format(k))
                                   for k in ks)
    else:
        raise ValueError

    baseline = BaselineStruct(link_alpha, prop_alpha, l1_ratio)

    all_Y_pred = []
    scores = []

    for k, (tr, val) in enumerate(KFold(n_folds).split(ids)):
        val_docs = list(_load("train", ids[val]))
        Y_true = []
        for doc in val_docs:
            y_prop = np.array([str(f['label_']) for f in doc.prop_features])
            y_link = np.array([f['label_'] for f in doc.features])
            Y_true.append(DocLabel(y_prop, y_link))

        X_tr_link, y_tr_link = load_csr(path.format(k, 'train.npz'),
                                        return_y=True)
        X_te_link, y_te_link = load_csr(path.format(k, 'val.npz'),
                                        return_y=True)

        X_tr_prop, y_tr_prop = load_csr(path.format(k, 'prop-train.npz'),
                                        return_y=True)
        X_te_prop, y_te_prop = load_csr(path.format(k, 'prop-val.npz'),
                                        return_y=True)

        baseline.fit(X_tr_link, y_tr_link, X_tr_prop, y_tr_prop)
        Y_marg = baseline.decision_function(X_te_link, X_te_prop, val_docs)

        zero_compat = np.zeros((baseline.n_prop_states, baseline.n_prop_states,
                                baseline.n_link_states))
        if decode:
            statuses = Counter()
            Y_pred = []
            for doc, y in zip(val_docs, Y_marg):
                doc.link_to_node_ = np.array(
                    [(f['src__prop_id_'], f['trg__prop_id_'])
                     for f in doc.features], dtype=np.intp)
                doc.second_order_ = []
                potentials = (y.nodes, y.links, zero_compat, [], [] ,[])
                y_decoded, status = baseline._inference(doc, potentials,
                                                        relaxed=False,
                                                        constraints=which)
                Y_pred.append(y_decoded)
                statuses[status] += 1

            logging.info("Test inference status: " +
                         ", ".join(
                             "{:.1f}% {}".format(100 * val / len(val_docs),
                                                 key)
                             for key, val in statuses.most_common()))
        else:
            Y_pred = [baseline._round(y.nodes, y.links, inverse_transform=True)
                      for y in Y_marg]
            all_Y_pred.extend(Y_pred)

        scores.append(baseline._score(Y_true, Y_pred))

    return scores, all_Y_pred


def saga_score_struct_cache(*args):

    arghash = sha1(repr(("score_struct",) + args).encode('utf-8')).hexdigest()
    fn = "res/baseline_linear_{}.dill".format(arghash)

    try:
        with open(fn, 'rb') as f:
            out = dill.load(f)
        logging.info("Loaded cached version.")
    except FileNotFoundError:
        logging.info("Computing...")
        out = saga_score_struct(*args)
        with open(fn, 'wb') as f:
            dill.dump(out, f)

    return out


def main():
    from docopt import docopt

    usage = """
    Usage:
        baseline_linear (cdcp|ukp) [--l1-ratio=N --decode]

    Options:
        --l1-ratio=N    amount of l1 [default: 0]
    """

    args = docopt(usage)

    which = 'cdcp' if args['cdcp'] else 'ukp'
    l1_ratio = float(args['--l1-ratio'])
    decode = args['--decode']

    n_alphas = 20
    alphas = np.logspace(-8, 0, n_alphas)

    link_scores, prop_scores = saga_cv_cache(which, alphas, l1_ratio)
    link_alpha_ix = np.argmax(link_scores.mean(axis=0))
    prop_alpha_ix = np.argmax(prop_scores.mean(axis=0))

    print("Link alpha={}".format(alphas[link_alpha_ix]))
    print("Prop alpha={}".format(alphas[prop_alpha_ix]))

    alpha_link = alphas[link_alpha_ix]
    alpha_prop = alphas[prop_alpha_ix]

    scores, _ = saga_score_struct_cache(which, alpha_link, alpha_prop,
                                        l1_ratio, decode)

    link_macro, link_micro, node_macro, node_micro, acc = np.mean(scores,
                                                                  axis=0)
    print("Link: {:.3f}/{:.3f} Node: {:.3f}/{:.3f} accuracy {:.3f}".format(
        link_macro, link_micro, node_macro, node_micro, acc))


if __name__ == '__main__':
    main()
