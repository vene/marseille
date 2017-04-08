# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

from collections import Counter
from operator import itemgetter
import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from nltk import Tree

from marseille.custom_logging import logging
from marseille.datasets import load_embeds
from marseille.features import add_pmi_features
from marseille.io import save_csr


class FilteredDictVectorizer(DictVectorizer):
    def __init__(self, columns, dtype=np.float64, sparse=True):
        super(FilteredDictVectorizer, self).__init__(dtype=dtype,
                                                     sparse=sparse)
        self.columns = columns

    def fit(self, X, y=None):
        return super(FilteredDictVectorizer, self).fit([
                                                           {key: x[key] for key
                                                            in self.columns}
                                                           for x in X
                                                           ])

    def fit_transform(self, X, y=None):
        return super(FilteredDictVectorizer, self).fit_transform([
                                                                     {key: x[
                                                                         key]
                                                                      for key
                                                                      in
                                                                      self.columns}
                                                                     for x in X
                                                                     ])

    def transform(self, X, y=None):
        return super(FilteredDictVectorizer, self).transform([
                                                                 {key: x[key]
                                                                  for key in
                                                                  self.columns}
                                                                 for x in X
                                                                 ])


def _lower_words_getter(feats):
    return (w.lower() for w in feats['words'])


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embeds, embed_vocab, mode='average'):
        self.embeds = embeds
        self.embed_vocab = embed_vocab
        self.mode = mode

    def fit(self, X, y=None, **fit_params):
        self.vect_ = TfidfVectorizer(vocabulary=self.embed_vocab,
                                     analyzer=_lower_words_getter,
                                     norm='l1',
                                     use_idf=False).fit(X)
        return self

    def transform(self, X, y=None):
        return safe_sparse_dot(self.vect_.transform(X), self.embeds)

    def fit_transform(self, X, y=None, **fit_params):
        self.vect_ = TfidfVectorizer(vocabulary=self.embed_vocab,
                                     analyzer=_lower_words_getter,
                                     norm='l1',
                                     use_idf=False)
        return safe_sparse_dot(self.vect_.fit_transform(X), self.embeds)

    def get_feature_names(self):
        return ["dim{:03d}".format(k)
                for k in range(self.embeds.shape[1])]


def custom_fnames(union):
    feature_names = []
    for name, trans, weight in union._iter():
        if hasattr(trans, 'get_feature_names'):
            this_fn = trans.get_feature_names()

        elif isinstance(trans, Pipeline):
            # we use pipelines to scale only specific attributes.
            # In this case, the vectorizer is first in the pipe.
            this_fn = trans.steps[0][-1].get_feature_names()

        else:
            raise AttributeError("Transformer %s (type %s) does not "
                                 "provide get_feature_names." % (
                                     str(name), type(trans).__name__))
        feature_names.extend([name + "__" + f for f in this_fn])
    return feature_names


class PrecedingStats(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.001):
        self.alpha = alpha

    def fit(self, feats):
        self.labels_ = ('MajorClaim', 'Claim', 'Premise')
        self.counts_ = {label: Counter() for label in self.labels_}
        for prop in feats:
            self.counts_[prop['label_']][tuple(prop['preceding_'])] += 1
        return self

    def count(self, preceding):
        return [self.counts_[label][preceding] for label in self.labels_]

    def transform(self, feats):
        stats = [self.count(tuple(prop['preceding_'])) for prop in feats]
        stats = np.array(stats, dtype=np.double)
        stats += self.alpha
        stats /= np.sum(stats, axis=1)[:, np.newaxis]

        return stats

    def get_feature_names(self):
        return ["p({}|prefix)".format(lbl) for lbl in self.labels_]


def stats_train(docs):
    lemma_freqs = Counter()
    prod_freqs = Counter()
    dep_freqs = Counter()

    count_outgoing = Counter()
    count_incoming = Counter()
    n_links = 0
    n_props = 0

    for doc in docs:
        lemma_freqs.update(w['lemma'].lower() for sent in doc.nlp['sentences']
                           for w in sent['tokens'])

        prod_freqs.update(
            str(prod) for sent in doc.nlp['sentences'] for prod in
            Tree.fromstring(sent['parse']).productions())

        for sent in doc.nlp['sentences']:
            lemmas = ['ROOT'] + [w['lemma'] for w in sent['tokens']]
            dep_freqs.update('{} -> {}'.format(lemmas[arc['dependent']],
                                               lemmas[arc['governor']])
                             for arc
                             in sent['collapsed-ccprocessed-dependencies'])

        n_props += len(doc.prop_offsets)

        for link in doc.features:
            if link['label_']:
                # tokens in src are counted as outgoing
                count_outgoing.update(w.lower() for w in link['src__lemmas'])
                n_links += 1
                # tokens in trg are counted as outgoing
                count_incoming.update(w.lower() for w in link['trg__lemmas'])

    link_log_proba = np.log(n_links) - np.log(n_props)

    def pmi(word, class_count):
        return (np.log(class_count[word]) - np.log(lemma_freqs[word]) -
                link_log_proba)

    pmi_incoming = {w: pmi(w, count_incoming) for w in count_incoming}
    pmi_outgoing = {w: pmi(w, count_outgoing) for w in count_outgoing}

    # ensure order
    lemma_freqs = sorted(list(lemma_freqs.most_common()),
                         key=lambda x: (-x[1], x[0]))
    prod_freqs = sorted(list(prod_freqs.most_common()),
                        key=lambda x: (-x[1], x[0]))
    dep_freqs = sorted(list(dep_freqs.most_common()),
                       key=lambda x: (-x[1], x[0]))

    return lemma_freqs, prod_freqs, dep_freqs, pmi_incoming, pmi_outgoing


def make_union_prop(vect_params):
    for key, params in sorted(vect_params.items()):
        yield key, CountVectorizer(analyzer=itemgetter(key), binary=True,
                                   **params)


def make_union_link(vect_params):
    for which in ('src', 'trg'):
        for key, params in sorted(vect_params.items()):
            getkey = "{}__{}".format(which, key)
            prefix = "[{}]_{}".format(which, key)
            vect_ = CountVectorizer(analyzer=itemgetter(getkey), binary=True,
                                    **params)
            yield prefix, vect_


def prop_vectorizer(train_docs, which, stats=None, n_most_common_tok=1000,
                    n_most_common_dep=1000, return_transf=False):
    # One pass to compute training corpus statistics.
    train_docs = list(train_docs)

    if stats is None:
        stats = stats_train(train_docs)
    lemma_freqs, _, dep_freqs, _, _ = stats

    # vectorize BOW-style features
    lemma_vocab = [w for w, _ in lemma_freqs[:n_most_common_tok]]
    dep_vocab = [p for p, _ in dep_freqs[:n_most_common_dep]]

    vects = dict(lemmas=dict(vocabulary=lemma_vocab, lowercase=True),
                 dependency_tuples=dict(vocabulary=dep_vocab), pos={},
                 discourse={}, indicators={}, indicator_preceding_in_para={},
                 indicator_following_in_para={})

    raw_keys = ['is_first_in_para', 'is_last_in_para', 'toks_to_sent_ratio',
                'relative_in_para', 'first_person_any', 'root_vb_modal',
                'root_vb_tense']
    nrm_keys = ['n_tokens', 'n_toks_in_sent', 'n_toks_in_para',
                'n_toks_preceding_in_sent', 'n_toks_following_in_sent',
                'preceding_props_in_para', 'following_props_in_para',
                'parse_tree_height', 'n_subordinate_clauses']
    if which == 'ukp':
        raw_keys += ['is_in_intro', 'is_in_conclusion',
                     'has_shared_np_intro', 'has_shared_vp_intro',
                     'has_shared_np_conclusion', 'has_shared_vp_conclusion']
        nrm_keys += ['n_shared_np_intro', 'n_shared_vp_intro',
                     'n_shared_np_conclusion', 'n_shared_vp_conclusion']

    # load embeds
    embed_vocab, embeds = load_embeds(which)

    vect_list = list(make_union_prop(vects)) + [
        ('raw', FilteredDictVectorizer(raw_keys)),
        ('nrm', make_pipeline(FilteredDictVectorizer(nrm_keys, sparse=False),
                              MinMaxScaler((0, 1)))),
        ('embeds', EmbeddingVectorizer(embeds, embed_vocab))]

    if which == 'ukp':
        vect_list.append(('proba', PrecedingStats()))

    vect = FeatureUnion(vect_list)

    train_feats = [f for doc in train_docs for f in doc.prop_features]

    if return_transf:
        X_tr = vect.fit_transform(train_feats)
        return vect, X_tr
    else:
        return vect.fit(train_feats)


def link_vectorizer(train_docs, stats=None, n_most_common=1000,
                    return_transf=False):
    # One pass to compute training corpus statistics.
    train_docs = list(train_docs)

    if stats is None:
        stats = stats_train(train_docs)
    lemma_freqs, prod_freqs, _, pmi_incoming, pmi_outgoing = stats

    # vectorize BOW-style features
    lemma_vocab = [w for w, _ in lemma_freqs[:n_most_common]]
    prod_vocab = [p for p, _ in prod_freqs[:n_most_common]]

    vects = dict(lemmas=dict(vocabulary=lemma_vocab, lowercase=True),
                 productions=dict(vocabulary=prod_vocab), pos={}, discourse={},
                 indicators={}, indicator_preceding_in_para={},
                 indicator_following_in_para={})

    raw_keys = ['src__is_first_in_para', 'src__is_last_in_para',
                'trg__is_first_in_para', 'trg__is_last_in_para',
                'same_sentence', 'src_precedes_trg', 'trg_precedes_src',
                'any_shared_nouns', 'src__pmi_pos_ratio', 'src__pmi_neg_ratio',
                'trg__pmi_pos_ratio', 'trg__pmi_neg_ratio', 'src__pmi_pos_any',
                'src__pmi_neg_any', 'trg__pmi_pos_any', 'trg__pmi_neg_any', ]
    nrm_keys = ['src__n_tokens', 'trg__n_tokens', 'props_between', 'n_props',
                'n_shared_nouns']

    vect_list = list(make_union_link(vects)) + [
        ('raw', FilteredDictVectorizer(raw_keys)), ('nrm', make_pipeline(
            FilteredDictVectorizer(nrm_keys, sparse=False),
            MinMaxScaler((0, 1))))]

    vect = FeatureUnion(vect_list)

    train_feats = [f for doc in train_docs for f in doc.features]
    [add_pmi_features(f, pmi_incoming, pmi_outgoing) for f in train_feats]

    if return_transf:
        X_tr = vect.fit_transform(train_feats)
        return vect, X_tr
    else:
        return vect.fit(train_feats)


def second_order_vectorizer(train_docs):
    # this is very simple and all features are already in 0,1

    train_docs = list(train_docs)

    raw_keys = ['same_sentence', 'same_sentence_ab', 'same_sentence_ac',
                'same_sentence_bc', 'order_abc', 'order_acb', 'order_bac',
                'order_bca', 'order_cab', 'order_cba', 'range_leq_1',
                'range_leq_2', 'range_leq_3', 'range_leq_4', 'range_leq_5',
                'any_shared_nouns', 'any_shared_nouns_ab',
                'any_shared_nouns_ac', 'any_shared_nouns_bc', 'jaccard',
                'jaccard_ab', 'jaccard_ac', 'jaccard_bc',
                'shared_nouns_ratio_a', 'shared_nouns_ratio_b',
                'shared_nouns_ratio_c', 'shared_nouns_ratio_ab',
                'shared_nouns_ratio_ac', 'shared_nouns_ratio_bc',
                'shared_nouns_ab_ratio_a', 'shared_nouns_ab_ratio_b',
                'shared_nouns_ac_ratio_a', 'shared_nouns_ac_ratio_c',
                'shared_nouns_bc_ratio_b', 'shared_nouns_bc_ratio_c']
    vect = FilteredDictVectorizer(raw_keys)

    vect.fit([f for doc in train_docs for f in doc.second_order_features])
    return vect


def vectorize(train_docs, test_docs, which, n_most_common=500):
    """Train a vectorizer on the training docs and transform the test docs.

    We use a function because scikit-learn vectorizers cannot change the
    number of samples, but we need to extract multiple rows from each doc.
    So we cannot use pipelines.
    """

    logging.info("Vectorizing...")

    # One pass to compute training corpus statistics.
    train_docs = list(train_docs)
    test_docs = list(test_docs)
    stats = stats_train(train_docs)
    _, _, _, pmi_incoming, pmi_outgoing = stats

    # link vectors
    vect, X_tr = link_vectorizer(train_docs, stats, n_most_common,
                                 return_transf=True)

    y_tr = np.array([f['label_'] for doc in train_docs for f in doc.features],
                    dtype=np.bool)

    test_feats = [f for doc in test_docs for f in doc.features]
    [add_pmi_features(f, pmi_incoming, pmi_outgoing) for f in test_feats]
    y_te = np.array([f['label_'] for f in test_feats], dtype=np.bool)

    X_te = vect.transform(test_feats)

    # prop vectors
    prop_vect, prop_X_tr = prop_vectorizer(train_docs, which, stats,
                                           n_most_common_tok=None,
                                           n_most_common_dep=2000,
                                           return_transf=True)
    prop_y_tr = np.array([str(f['label_']) for doc in train_docs
                          for f in doc.prop_features])
    prop_y_te = np.array([str(f['label_']) for doc in test_docs
                          for f in doc.prop_features])
    test_feats = [f for doc in test_docs for f in doc.prop_features]
    prop_X_te = prop_vect.transform(test_feats)

    return ((prop_X_tr, prop_X_te, prop_y_tr, prop_y_te, prop_vect),
            (X_tr, X_te, y_tr, y_te, vect))


def main():
    from marseille.argdoc import CdcpArgumentationDoc, UkpEssayArgumentationDoc
    from marseille.datasets import (cdcp_train_ids, cdcp_test_ids,
                                    ukp_train_ids, ukp_test_ids)
    from docopt import docopt

    usage = """
    Usage:
        vectorize folds (cdcp|ukp) [--n-folds=N]
        vectorize split (cdcp|ukp)

    Options:
        --n-folds=N  number of cross-val folds to generate. [default: 3]
    """

    args = docopt(usage)

    which = 'ukp' if args['ukp'] else 'cdcp' if args['cdcp'] else None
    if args['ukp']:
        _tpl = os.path.join("data", "process", "ukp-essays", "essay{:03d}")
        _path = os.path.join("data", "process", "ukp-essays", "folds", "{}",
                             "{}")
        _load = lambda which, ks: (UkpEssayArgumentationDoc(_tpl.format(k))
                                   for k in ks)
        ids = ukp_train_ids
        test_ids = ukp_test_ids

    else:
        _tpl = os.path.join("data", "process", "erule", "{}", "{:05d}")
        _path = os.path.join("data", "process", "erule", "folds", "{}", "{}")

        _load = lambda which, ks: (CdcpArgumentationDoc(_tpl.format(which, k))
                                   for k in ks)
        ids = cdcp_train_ids
        test_ids = cdcp_test_ids

    if args['folds']:

        n_folds = int(args['--n-folds'])
        ids = np.array(ids)

        for k, (tr, val) in enumerate(KFold(n_folds).split(ids)):
            train_docs = _load("train", ids[tr])
            val_docs = _load("train", ids[val])

            prop_out, link_out = vectorize(train_docs, val_docs,
                                           which=which)
            X_tr, X_val, y_tr, y_val, vect = link_out

            fnames = custom_fnames(vect)

            save_csr(_path.format(k, "train.npz"), X_tr, y_tr)
            save_csr(_path.format(k, "val.npz"), X_val, y_val)
            with open(_path.format(k, "fnames.txt"), "w") as f:
                for fname in fnames:
                    print(fname, file=f)

            X_tr, X_val, y_tr, y_val, vect = prop_out
            fnames = custom_fnames(vect)

            save_csr(_path.format(k, "prop-train.npz"), X_tr, y_tr)
            save_csr(_path.format(k, "prop-val.npz"), X_val, y_val)
            with open(_path.format(k, "prop-fnames.txt"), "w") as f:
                for fname in fnames:
                    print(fname, file=f)

    elif args['split']:
        train_docs = _load("train", ids)
        test_docs = _load("test", test_ids)

        prop_out, link_out = vectorize(train_docs, test_docs, which=which)

        X_tr, X_te, y_tr, y_te, vect = link_out
        fnames = custom_fnames(vect)
        save_csr(_path.format("traintest", "train.npz"), X_tr, y_tr)
        save_csr(_path.format("traintest", "test.npz"), X_te, y_te)
        with open(_path.format("traintest", "fnames.txt"), "w") as f:
            for fname in fnames:
                print(fname, file=f)

        X_tr, X_te, y_tr, y_te, vect = prop_out
        fnames = custom_fnames(vect)
        save_csr(_path.format("traintest", "prop-train.npz"), X_tr, y_tr)
        save_csr(_path.format("traintest", "prop-test.npz"), X_te, y_te)
        with open(_path.format("traintest", "prop-fnames.txt"), "w") as f:
            for fname in fnames:
                print(fname, file=f)

if __name__ == '__main__':
    main()
