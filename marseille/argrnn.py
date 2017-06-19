"""Argumentation structure RNN using dynet and AD3 inference"""

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

import warnings
from collections import Counter
from time import time

import numpy as np
import dynet as dy

from sklearn.utils import shuffle

from marseille.struct_models import BaseArgumentMixin
from marseille.inference import loss_augment_unaries
from marseille.custom_logging import logging
from marseille.dynet_utils import (MultiLayerPerceptron, Bilinear,
                                 MultilinearFactored)


class LinkMLP(dy.Saveable):
    """Argumentation link encoder with a multi-layer perceptron:

    Performs MLP(concat(src, trg)"""

    def __init__(self, n_in, n_hid, n_out, n_layers, model):
        self.mlp = MultiLayerPerceptron(
            [2 * n_in] + [n_hid] * n_layers + [n_out], activation=dy.rectify,
            model=model)

    def __call__(self, src, trg):
        return self.mlp(dy.concatenate([src, trg]))

    def get_components(self):
        return [self.mlp]

    def restore_components(self, components):
        self.mlp = components[0]

    def set_dropout(self, dropout):
        self.mlp.dropout = dropout


class LinkBilinear(dy.Saveable):
    """Bilinear argumentation link encoder:

    src_t = MLP_src(src)
    trg_t = MLP_trg(trg)
    return src_t' * W * trg_t + linear and bias terms

    """
    def __init__(self, n_in, n_hid, n_out, n_layers, model):
        dims = [n_in] + [n_hid] * n_layers,
        self.src_mlp = MultiLayerPerceptron(dims, activation=dy.rectify,
                                             model=model)
        self.trg_mlp = MultiLayerPerceptron(dims, activation=dy.rectify,
                                             model=model)

        self.bilinear = Bilinear(n_hid, n_out, model=model)

    def __call__(self, src, trg):
        return self.bilinear(
            dy.rectify(self.src_mlp(src)),  # HOTFIX rectify here?
            dy.rectify(self.trg_mlp(trg)))

    def get_components(self):
        return [self.src_mlp, self.trg_mlp, self.bilinear]

    def restore_components(self, components):
        self.src_mlp, self.trg_mlp, self.bilinear = components

    def set_dropout(self, dropout):
        self.src_mlp.dropout = dropout
        self.trg_mlp.dropout = dropout


class SecondOrderMLP(dy.Saveable):
    """Second-order encoders using multi-layer perceptrons:

    phi(a, b, c) = MLP(concat(a, b, c))
    """
    def __init__(self, n_in, n_hid, n_layers, model):
        second_order_dims = [3 * n_in] + [n_hid] * n_layers + [1]
        self.mlp = MultiLayerPerceptron(second_order_dims,
                                        activation=dy.rectify,
                                        model=model)

    def __call__(self, a, b, c):
        return self.mlp(dy.concatenate([a, b, c]))

    def get_components(self):
        return [self.mlp]

    def restore_components(self, components):
        self.mlp = components[0]

    def set_dropout(self, dropout):
        self.mlp.dropout = dropout


class SecondOrderMultilinear(dy.Saveable):
    """Second-order encoder using low-rank multilinear term:

    a_t, b_t, c_t = MLP_a(a), MLP_b(b), MLP_c(c)
    w_ijk = sum_0<r<rank u^(a)_ir u^(b)_jr u^(c)_kr
    phi(a, b, c) = sum_ijk a_i b_j c_k w_ijk
    """
    def __init__(self, n_in, n_hid, n_layers, model, n_components=16):
        dims = [n_in] + [n_hid] * n_layers
        self.a_mlp = MultiLayerPerceptron(dims, activation=dy.rectify,
                                          model=model)
        self.b_mlp = MultiLayerPerceptron(dims, activation=dy.rectify,
                                          model=model)
        self.c_mlp = MultiLayerPerceptron(dims, activation=dy.rectify,
                                          model=model)
        self.multilinear = MultilinearFactored(n_features=1 + n_hid,
                                               n_inputs=3,
                                               n_components=n_components,
                                               model=model)

    def __call__(self, a, b, c):
        enc = [dy.rectify(self.a_mlp(a)),  # HOTFIX rectify here?
               dy.rectify(self.b_mlp(b)),
               dy.rectify(self.c_mlp(c))]
        enc = [dy.concatenate([dy.scalarInput(1), x]) for x in enc]
        return self.multilinear(*enc)

    def get_components(self):
        return [self.a_mlp, self.b_mlp, self.c_mlp, self.multilinear]

    def restore_components(self, components):
        self.a_mlp, self.b_mlp, self.c_mlp, self.multilinear = components

    def set_dropout(self, dropout):
        self.a_mlp.dropout = dropout
        self.b_mlp.dropout = dropout
        self.c_mlp.dropout = dropout


class ArgumentLSTM(BaseArgumentMixin):
    """Argumentation parser using LSTM features and structured hinge loss.

    Parameters
    ----------

    max_iter: int, default: 100
        Total number of iterations (epochs) to perform.

    score_at_iter: list or None, default: None
        Number of iterations after which to compute scores on validation data.

    n_embed: int, default: 128
        Embedding size. Ignored if existing embeddings passed to `embeds`.

    lstm_layers: int, default: 2
        Number of LSTM layers to use.

    prop_mlp_layers: int, default: 2
        Number of layers in proposition MLP encoder.

    link_mlp_layers: int, default: 1
        Number of layers in link encoder (either as MLP or as preprocessing
        before the bilinear calculation.)

    link_bilinear: bool, default: True
        Whether to use bilinear model for encoding link potentials.

    n_lstm: int, default: 128
        LSTM hidden layer size. (Must be even: is split into forward and
        backward LSTMs equally.)

    n_mlp: int, default: 128
        Hidden layer size across all MLPs.

    lstm_dropout: float, default 0
        Amount of LSTM dropout. Might be buggy in dynet at the moment.

    mlp_dropout: float, default: 0.1
        Amount of dropout to apply across all MLPs in the model.

    embeds: tuple or None, default: None
        Tuple of (embed_data, embed_vocab) for GloVe initialization

    class_weight: "balanced" or None, default: "balanced"
        Scaling for the negative link class cost. Does not influence
        proposition loss.

    exact_inference: bool, default: False
        Whether to use branch & bound at every iteration to get exact
        solutions. Can be very slow.

    compat_features: bool, default: False
        Whether to use structural features to parametrize the compatibility
        factors.  Documents should be preprocessed accordingly.

    constraints: {"ukp"|"ukp+strict"|"cdcp"|"cdcp+strict"|None}, default: None
        What kind of constraints to apply in decoding.

    second_order_multilinear: bool, default: True
        Whether to use low-rank multilinear encoder for second order
        potentials (only used if at least one of
        (coparent|grandparent|sibling)_layers is nonzero.

    coparent_layers: int, default: 0
        Number of layers to use in coparent potential encoding. If 0,
        coparents are not used.

    grandparent_layers: int, default: 0
        Number of layers to use in grandparent potential encoding. If 0,
        grandparent are not used.

    sibling_layers: int, default: 0
        Number of layers to use in sibling potential encoding. If 0,
        sibling are not used.

    multilinear_rank: int, default: 16
        Rank of the third-order tensor for coparent, grandparent and sibling
        potentials. Only used if at least one of such potentials is on, and if
        `second_order_multilinear=True`.

    """
    def __init__(self, max_iter=100, score_at_iter=None, n_embed=128,
                 lstm_layers=2, prop_mlp_layers=2, link_mlp_layers=1,
                 link_bilinear=True, n_lstm=128, n_mlp=128,
                 lstm_dropout=0.0, mlp_dropout=0.1, embeds=None,
                 class_weight=None, exact_inference=False,
                 compat_features=False, constraints=None,
                 second_order_multilinear=True, coparent_layers=0,
                 grandparent_layers=0, sibling_layers=0,
                 exact_test=False):
        self.max_iter = max_iter
        self.score_at_iter = score_at_iter
        self.n_embed = n_embed
        self.lstm_layers = lstm_layers
        self.prop_mlp_layers = prop_mlp_layers
        self.link_mlp_layers = link_mlp_layers
        self.link_bilinear = link_bilinear
        self.n_lstm = n_lstm
        self.n_mlp = n_mlp
        self.lstm_dropout = lstm_dropout
        self.mlp_dropout = mlp_dropout
        self.embeds = embeds
        self.class_weight = class_weight
        self.exact_inference = exact_inference
        self.compat_features = compat_features
        self.constraints = constraints
        self.second_order_multilinear = second_order_multilinear
        self.coparent_layers = coparent_layers
        self.grandparent_layers = grandparent_layers
        self.sibling_layers = sibling_layers
        self.exact_test = exact_test

    def build_vocab(self, docs):
        special_toks = ["__UNK__"]
        word_df = Counter(tok for doc in docs for tok in set(doc.tokens()))
        vocab = special_toks + [w for w, df in sorted(word_df.items())
                                if df > 1]
        inv_vocab = {word: k for k, word in enumerate(vocab)}
        self.UNK = inv_vocab["__UNK__"]
        self.inv_vocab = inv_vocab
        self.vocab = vocab

    def init_params(self):
        self.model = dy.Model()
        self._trainer = dy.AdamTrainer(self.model)

        if self.embeds is not None:
            sz = self.embeds[1].shape[1]
            self.n_embed = sz
            logging.info("Overriding n_embeds to glove size {}".format(sz))

        self._embed = self.model.add_lookup_parameters((len(self.vocab),
                                                        self.n_embed))

        if self.embeds is not None:  # initialize embeddings with glove
            logging.info("Initializing embeddings...")
            embed_vocab, embed_data = self.embeds
            inv_embed = {w: k for k, w in enumerate(embed_vocab)}
            for k, w in enumerate(self.vocab):
                if w in inv_embed:
                    self._embed.init_row(k, embed_data[inv_embed[w]])
            logging.info("...done")

        self._rnn = dy.BiRNNBuilder(self.lstm_layers, self.n_embed,
                                    self.n_lstm, self.model, dy.LSTMBuilder)

        # proposition classifier MLP
        self._prop_mlp = MultiLayerPerceptron(
            [self.n_lstm] +
            [self.n_mlp] * self.prop_mlp_layers +
            [self.n_prop_states],
            activation=dy.rectify,
            model=self.model)

        # link classifier MLP (possibly bilinear)
        LinkEncoder = LinkBilinear if self.link_bilinear else LinkMLP
        self._link = LinkEncoder(self.n_lstm, self.n_mlp, self.n_link_states,
                                 self.link_mlp_layers, self.model)

        # compatibility (trigram) factors, optionally with features
        n_compat = self.n_prop_states ** 2 * self.n_link_states
        if self.compat_features:
            n_compat *= self.n_compat_features

        self._compat = self.model.add_parameters(n_compat,
                                                 init=dy.ConstInitializer(0))

        # optional second-order scorers
        SecondOrderEncoder = (SecondOrderMultilinear
                              if self.second_order_multilinear
                              else SecondOrderMLP)

        if self.coparent_layers:  # scorer for a -> b <- c
            self._coparent = SecondOrderEncoder(self.n_lstm, self.n_mlp,
                                                self.coparent_layers,
                                                self.model)

        if self.grandparent_layers:  # scorer for a -> b -> c
            self._grandparent = SecondOrderEncoder(self.n_lstm, self.n_mlp,
                                                   self.grandparent_layers,
                                                   self.model)

        if self.sibling_layers:  # scorer for a <- b -> c
            self._sibling = SecondOrderEncoder(self.n_lstm, self.n_mlp,
                                               self.sibling_layers,
                                               self.model)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()
                if k != 'model' and k[0] != '_'}

    def save(self, filename):
        params = [self._compat, self._embed, self._rnn, self._prop_mlp,
                  self._link]

        if self.coparent_layers:
            params.extend([self._coparent])

        if self.grandparent_layers:
            params.extend([self._grandparent])

        if self.sibling_layers:
            params.extend([self._sibling])

        self.model.save(filename, params)

    def load(self, filename):
        self.init_params()
        saved = self.model.load(filename)
        (self._compat, self._embed, self._rnn, self._prop_mlp,
         self._link) = saved[:5]
        saved = saved[5:]
        saved.reverse()  # so we can just pop

        if self.coparent_layers:
            self._coparent = saved.pop()
        if self.grandparent_layers:
            self._grandparent = saved.pop()
        if self.sibling_layers:
            self._sibling = saved.pop()

        assert len(saved) == 0

    def build_cg(self, doc, training=True):

        dy.renew_cg()

        # dropout
        if training:
            self._rnn.set_dropout(self.lstm_dropout)
            drop = self.mlp_dropout
        else:
            self._rnn.disable_dropout()
            drop = 0

        self._prop_mlp.dropout = drop
        self._link.set_dropout(drop)
        if self.sibling_layers:
            self._sibling.set_dropout(drop)
        if self.grandparent_layers:
            self._grandparent.set_dropout(drop)
        if self.coparent_layers:
            self._coparent.set_dropout(drop)

        # lookup token embeddings
        tok_ids = (self.inv_vocab.get(tok, self.UNK) for tok in doc.tokens())
        embeds = [dy.lookup(self._embed, tok) for tok in tok_ids]

        # pass through bidi LSTM
        rnn_out = self._rnn.transduce(embeds)

        # map character offsets to token offsets
        tok_offset = np.array(doc.tokens(key='characterOffsetBegin',
                                         lower=False))

        # get an average representation for each proposition
        prop_repr = []
        prop_potentials = []
        link_potentials = []
        coparent_potentials = []
        grandparent_potentials = []
        sibling_potentials = []

        for offset in doc.prop_offsets:
            start, end = np.searchsorted(tok_offset, offset)
            prop = dy.average(rnn_out[start:end])
            prop_potentials.append(self._prop_mlp(prop))
            prop_repr.append(prop)

        for src, trg in doc.link_to_prop:
            link_potentials.append(self._link(prop_repr[src], prop_repr[trg]))

        # optional second order factor scores
        any_second_order = (self.coparent_layers or
                            self.grandparent_layers or
                            self.sibling_layers)

        if any_second_order:
            for a, b, c in doc.second_order:
                x = (prop_repr[a], prop_repr[b], prop_repr[c])

                if self.coparent_layers:
                    coparent_potentials.append(self._coparent(*x))

                if self.grandparent_layers:
                    grandparent_potentials.append(self._grandparent(*x))

                if self.sibling_layers:
                    sibling_potentials.append(self._sibling(*x))

        compat = dy.parameter(self._compat)

        return (prop_potentials,
                link_potentials,
                compat,
                coparent_potentials,
                grandparent_potentials,
                sibling_potentials)

    def _get_potentials(self, doc, dy_potentials):

        props, links, compat, coparents, grandpas, siblings = dy_potentials

        prop_potentials = dy.concatenate_cols(props)
        prop_potentials = prop_potentials.value().astype(np.double).T
        link_potentials = dy.concatenate_cols(links)
        link_potentials = link_potentials.value().astype(np.double).T

        if self.compat_features:
            w_compat = compat.npvalue().reshape(self.n_compat_features, -1)
            compat_potentials = (np.dot(doc.compat_features, w_compat)
                                 .reshape(-1,
                                          self.n_prop_states,
                                          self.n_prop_states,
                                          self.n_link_states))
        else:
            compat_potentials = compat.npvalue().reshape(self.n_prop_states,
                                                         self.n_prop_states,
                                                         self.n_link_states)

        coparent_potentials = (dy.concatenate(coparents).value()
                               if coparents else [])
        grandparent_potentials = (dy.concatenate(grandpas).value()
                                  if grandpas else [])
        sibling_potentials = (dy.concatenate(siblings).value()
                              if siblings else [])

        return (prop_potentials, link_potentials, compat_potentials,
                coparent_potentials, grandparent_potentials,
                sibling_potentials)

    def _doc_loss(self, doc, y):

        dy_potentials = self.build_cg(doc)
        potentials = self._get_potentials(doc, dy_potentials)

        # unpack all potentials

        (dy_prop_potentials,
         dy_link_potentials,
         dy_compat_potentials,
         dy_coparent_potentials,
         dy_grandparent_potentials,
         dy_sibling_potentials) = dy_potentials

        (prop_potentials,
         link_potentials,
         compat_potentials,
         coparent_potentials,
         grandparent_potentials,
         sibling_potentials) = potentials

        y_prop = self.prop_encoder_.transform(y.nodes)
        y_link = self.link_encoder_.transform(y.links)

        loss_augment_unaries(prop_potentials, y_prop, self.prop_cw_)
        loss_augment_unaries(link_potentials, y_link, self.link_cw_)

        y_hat, status = self._inference(doc, potentials, relaxed=True,
                                        exact=self.exact_inference,
                                        constraints=self.constraints)
        (prop_marg,
         link_marg,
         compat_marg,
         second_order_marg) = self._marg_fractional(doc, y_hat)

        (Y_prop,
         Y_link,
         compat_true,
         second_order_true) = self._marg_rounded(doc, y)

        # proposition loss
        prop_ix = np.arange(len(y_prop))
        prop_cws = self.prop_cw_[y_prop]
        prop_hamm = prop_cws * (1 - prop_marg[prop_ix, y_prop])
        diffs = prop_marg - Y_prop
        obj_prop = [dy.dot_product(prop, dy.inputVector(diff))
                    for prop, diff, hamm
                    in zip(dy_prop_potentials, diffs, prop_hamm)
                    if hamm > 1e-9]

        # link loss
        link_ix = np.arange(len(y_link))
        link_cws = self.link_cw_[y_link]
        link_hamm = link_cws * (1 - link_marg[link_ix, y_link])
        diffs = link_marg - Y_link
        obj_link = [dy.dot_product(link, dy.inputVector(diff))
                    for link, diff, hamm
                    in zip(dy_link_potentials, diffs, link_hamm)
                    if hamm > 1e-9]

        hamming_loss = prop_hamm.sum() + link_hamm.sum()
        max_hamming_loss = prop_cws.sum() + link_cws.sum()
        obj = obj_prop + obj_link

        # append compat objective
        compat_diff = (compat_marg - compat_true).ravel()
        compat_obj = dy.dot_product(dy_compat_potentials,
                                    dy.inputVector(compat_diff))
        obj.append(compat_obj)

        # append second order objective
        second_order_potentials = (dy_coparent_potentials +
                                   dy_grandparent_potentials +
                                   dy_sibling_potentials)

        if second_order_potentials:
            second_order_potentials = dy.concatenate(second_order_potentials)
            second_order_diff = second_order_marg - second_order_true
            second_order_obj = dy.dot_product(second_order_potentials,
                                             dy.inputVector(second_order_diff))
            second_order_obj = second_order_obj
            obj.append(second_order_obj)

        obj = dy.esum(obj)

        return obj, hamming_loss, max_hamming_loss, status

    def initialize(self, docs, Y):

        if self.compat_features:
            self.n_compat_features = docs[0].compat_features.shape[1]

        self.coparents_ = self.coparent_layers > 0
        self.grandparents_ = self.grandparent_layers > 0
        self.siblings_ = self.sibling_layers > 0

        self.build_vocab(docs)
        self.initialize_labels(Y)

        self.init_params()

    def fit(self, docs, Y, docs_val=None, Y_val=None):

        self.initialize(docs, Y)

        self.scores_ = []
        if self.score_at_iter:
            score_at_iter = self.score_at_iter
        else:
            score_at_iter = []

        train_time = 0
        for it in range(self.max_iter):
            # evaluate
            if docs_val and it in score_at_iter:
                Y_val_pred = self.predict(docs_val, exact=False)
                val_scores = self._score(Y_val, Y_val_pred)
                self.scores_.append(val_scores)

                with warnings.catch_warnings() as w:
                    warnings.simplefilter('ignore')
                    print("\t\t   val link: {:.3f}/{:.3f} Node: {:.3f}/{:.3f} "
                          "accuracy {:.3f}".format(*val_scores))

            docs, Y = shuffle(docs, Y, random_state=0)

            iter_loss = 0
            iter_max_loss = 0
            inference_status = Counter()

            tic = time()
            for doc, y in zip(docs, Y):
                if len(y.nodes) == 0:
                    continue

                obj, loss, max_loss, status = self._doc_loss(doc, y)
                inference_status[status] += 1

                iter_loss += loss
                iter_max_loss += max_loss

                if loss < 1e-9:
                    continue

                obj.scalar_value()
                obj.backward()
                self._trainer.update()

            self._trainer.update_epoch()
            self._trainer.status()
            toc = time()
            train_time += toc - tic
            print("Iter {} loss {:.4f}".format(it, iter_loss / iter_max_loss))
            print(", ".join("{:.1f}% {}".format(100 * val / len(docs), key)
                            for key, val in inference_status.most_common()))

            if iter_loss < 1e-9:
                break

        if docs_val and self.max_iter in score_at_iter:
            Y_val_pred = self.predict(docs_val, exact=False)
            val_scores = self._score(Y_val, Y_val_pred)
            self.scores_.append(val_scores)

        logging.info("Training time: {:.2f}s/iteration ({:.2f}s/doc-iter)"
            .format(train_time / it, train_time / (it * len(docs))))

    def predict(self, docs, exact=None):
        if exact is None:
            exact = self.exact_test

        pred = []
        statuses = Counter()
        tic = time()
        for doc in docs:
            dy_potentials = self.build_cg(doc, training=False)
            potentials = self._get_potentials(doc, dy_potentials)
            y_pred, status = self._inference(doc, potentials, relaxed=False,
                                             exact=exact,
                                             constraints=self.constraints)
            pred.append(y_pred)
            statuses[status] += 1
        toc = time()
        logging.info("Prediction time: {:.2f}s/doc".format((toc - tic) /
                                                           len(docs)))
        logging.info("Test inference status: " +
                     ", ".join(
                         "{:.1f}% {}".format(100 * val / len(docs), key)
                         for key, val in statuses.most_common()))
        return pred


class BaselineArgumentLSTM(ArgumentLSTM):
    """Baseline multi-task model with only proposition and link potentials.

    Evaluation is done both directly (argmax) or after decoding with
    constraints, depending on whether constraints == None.
    For code simplicity, AD3 is still called, but decoding should be
    instantaneous without extra constraints.)
    """

    def __init__(self, max_iter=100, score_at_iter=None, n_embed=128,
                 lstm_layers=2, prop_mlp_layers=2, link_mlp_layers=1,
                 link_bilinear=True, n_lstm=128, n_mlp=128,
                 lstm_dropout=0.0, mlp_dropout=0.1, embeds=None,
                 exact_inference=False, constraints=None,
                 exact_test=False):
        super(BaselineArgumentLSTM, self).__init__(
            max_iter=max_iter, score_at_iter=score_at_iter,
            n_embed=n_embed, lstm_layers=lstm_layers,
            prop_mlp_layers=prop_mlp_layers,
            link_mlp_layers=link_mlp_layers, link_bilinear=link_bilinear,
            n_lstm=n_lstm, n_mlp=n_mlp, lstm_dropout=lstm_dropout,
            mlp_dropout=mlp_dropout, embeds=embeds, class_weight=None,
            exact_inference=exact_inference, compat_features=False,
            constraints=constraints, second_order_multilinear=False,
            coparent_layers=0, grandparent_layers=0, sibling_layers=0,
            exact_test=exact_test)

    def _doc_loss(self, doc, y):
        y_node = self.prop_encoder_.transform(y.nodes)
        y_link = self.link_encoder_.transform(y.links)

        props, links, _, _, _, _ = self.build_cg(doc)

        obj_prop = [dy.hinge(prop, y_) for prop, y_ in zip(props, y_node)]
        obj_link = [dy.hinge(link, y_) for link, y_ in zip(links, y_link)]

        obj = dy.esum(obj_prop) + dy.esum(obj_link)

        correct = sum(1 for val in obj_prop + obj_link
                      if val.scalar_value() == 0)

        max_acc = len(obj_prop + obj_link)
        return obj, max_acc - correct, max_acc, 'n/a'

