"""
Pystruct-compatible models.
"""

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

# AD3 is (c) Andre F. T. Martins, LGPLv3.0: http://www.cs.cmu.edu/~ark/AD3/

import sys
import warnings
import numpy as np

from sklearn.utils import compute_class_weight
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from pystruct.models import StructuredModel

from marseille.inference import loss_augment_unaries, CDCP_ILLEGAL_LINKS
from marseille.vectorize import add_pmi_features
from marseille.argdoc import DocStructure, DocLabel
from marseille.custom_logging import logging

from itertools import permutations

from ad3 import factor_graph as fg


def _binary_2d(y):
    if y.shape[1] == 1:
        y = np.column_stack([1 - y, y])
    return y


def arg_f1_scores(Y_true, Y_pred, **kwargs):

    macro = []
    micro_true = []
    micro_pred = []

    for y_true, y_pred in zip(Y_true, Y_pred):
        macro.append(f1_score(y_true, y_pred, **kwargs))
        micro_true.extend(y_true)
        micro_pred.extend(y_pred)

    return np.mean(macro), f1_score(micro_true, micro_pred, **kwargs)


class BaseArgumentMixin(object):

    def initialize_labels(self, Y):

        y_nodes_flat = [y_val for y in Y for y_val in y.nodes]
        y_links_flat = [y_val for y in Y for y_val in y.links]
        self.prop_encoder_ = LabelEncoder().fit(y_nodes_flat)
        self.link_encoder_ = LabelEncoder().fit(y_links_flat)

        self.n_prop_states = len(self.prop_encoder_.classes_)
        self.n_link_states = len(self.link_encoder_.classes_)

        self.prop_cw_ = np.ones_like(self.prop_encoder_.classes_,
                                     dtype=np.double)
        self.link_cw_ = compute_class_weight(self.class_weight,
                                             self.link_encoder_.classes_,
                                             y_links_flat)

        self.link_cw_ /= self.link_cw_.min()


        logging.info('Setting node class weights {}'.format(", ".join(
            "{}: {}".format(lbl, cw) for lbl, cw in zip(
                self.prop_encoder_.classes_, self.prop_cw_))))

        logging.info('Setting link class weights {}'.format(", ".join(
            "{}: {}".format(lbl, cw) for lbl, cw in zip(
                self.link_encoder_.classes_, self.link_cw_))))

    def _round(self, prop_marg, link_marg, prop_unary=None, link_unary=None,
               inverse_transform=True):

        # ensure ties are broken according to unary scores
        if prop_unary is not None:
            prop_unary = prop_unary.copy()
            prop_unary -= np.min(prop_unary)
            prop_unary /= np.max(prop_unary) * np.max(prop_marg)
            prop_marg[prop_marg > 1e-9] += prop_unary[prop_marg > 1e-9]
        if link_unary is not None:
            link_unary = link_unary.copy()
            link_unary -= np.min(link_unary)
            link_unary /= np.max(link_unary) * np.max(link_marg)
            link_marg[link_marg > 1e-9] += link_unary[link_marg > 1e-9]

        y_hat_props = np.argmax(prop_marg, axis=1)
        y_hat_links = np.argmax(link_marg, axis=1)
        if inverse_transform:
            y_hat_props = self.prop_encoder_.inverse_transform(y_hat_props)
            y_hat_links = self.link_encoder_.inverse_transform(y_hat_links)
        return DocLabel(y_hat_props, y_hat_links)

    def loss(self, y, y_hat):

        if not isinstance(y_hat, DocLabel):
            return self.continuous_loss(y, y_hat)

        y_nodes = self.prop_encoder_.transform(y.nodes)
        y_links = self.link_encoder_.transform(y.links)

        node_loss = np.sum(self.prop_cw_[y_nodes] * (y.nodes != y_hat.nodes))
        link_loss = np.sum(self.link_cw_[y_links] * (y.links != y_hat.links))

        return node_loss + link_loss

    def max_loss(self, y):
        y_nodes = self.prop_encoder_.transform(y.nodes)
        y_links = self.link_encoder_.transform(y.links)
        return np.sum(self.prop_cw_[y_nodes]) + np.sum(self.link_cw_[y_links])

    def continuous_loss(self, y, y_hat):

        if isinstance(y_hat, DocLabel):
            raise ValueError("continuous loss on discrete input")

        if isinstance(y_hat[0], tuple):
            y_hat = y_hat[0]

        prop_marg, link_marg = y_hat
        y_nodes = self.prop_encoder_.transform(y.nodes)
        y_links = self.link_encoder_.transform(y.links)

        prop_ix = np.indices(y.nodes.shape)
        link_ix = np.indices(y.links.shape)

        # relies on prop_marg and link_marg summing to 1 row-wise
        prop_loss = np.sum(self.prop_cw_[y_nodes] *
                           (1 - prop_marg[prop_ix, y_nodes]))

        link_loss = np.sum(self.link_cw_[y_links] *
                           (1 - link_marg[link_ix, y_links]))

        loss = prop_loss + link_loss
        return loss

    def _marg_rounded(self, x, y):
        y_node = y.nodes
        y_link = y.links
        Y_node = label_binarize(y_node, self.prop_encoder_.classes_)
        Y_link = label_binarize(y_link, self.link_encoder_.classes_)

        # XXX can this be avoided?
        Y_node, Y_link = map(_binary_2d, (Y_node, Y_link))

        src_type = Y_node[x.link_to_prop[:, 0]]
        trg_type = Y_node[x.link_to_prop[:, 1]]

        if self.compat_features:
            pw = np.einsum('...j,...k,...l->...jkl',
                           src_type, trg_type, Y_link)
            compat = np.tensordot(x.X_compat.T, pw, axes=[1, 0])
        else:
            # equivalent to compat_features == np.ones(n_links)
            compat = np.einsum('ij,ik,il->jkl', src_type, trg_type, Y_link)

        second_order = []

        if self.coparents_ or self.grandparents_ or self.siblings_:
            link = {(a, b): k for k, (a, b) in enumerate(x.link_to_prop)}
            if self.coparents_:
                second_order.extend(y_link[link[a, b]] & y_link[link[c, b]]
                                   for a, b, c in x.second_order)
            if self.grandparents_:
                second_order.extend(y_link[link[a, b]] & y_link[link[b, c]]
                                   for a, b, c in x.second_order)
            if self.siblings_:
                second_order.extend(y_link[link[b, a]] & y_link[link[b, c]]
                                   for a, b, c in x.second_order)
        second_order = np.array(second_order)

        return Y_node, Y_link, compat, second_order

    def _marg_fractional(self, x, y):
        (prop_marg, link_marg), (compat_marg, second_order_marg) = y

        if self.compat_features:
            compat_marg = np.tensordot(x.X_compat.T, compat_marg, axes=[1, 0])
        else:
            compat_marg = compat_marg.sum(axis=0)

        return prop_marg, link_marg, compat_marg, second_order_marg

    def _inference(self, x, potentials, exact=False, relaxed=True,
                   return_energy=False, constraints=None,
                   eta=0.1, adapt=True, max_iter=5000,
                   verbose=False):

        (prop_potentials,
         link_potentials,
         compat_potentials,
         coparent_potentials,
         grandparent_potentials,
         sibling_potentials) = potentials

        n_props, n_prop_classes = prop_potentials.shape
        n_links, n_link_classes = link_potentials.shape

        g = fg.PFactorGraph()
        g.set_verbosity(verbose)

        prop_vars = [g.create_multi_variable(n_prop_classes)
                     for _ in range(n_props)]
        link_vars = [g.create_multi_variable(n_link_classes)
                     for _ in range(n_links)]

        for var, scores in zip(prop_vars, prop_potentials):
            for state, score in enumerate(scores):
                var.set_log_potential(state, score)

        for var, scores in zip(link_vars, link_potentials):
            for state, score in enumerate(scores):
                var.set_log_potential(state, score)

        # compatibility trigram factors
        compat_factors = []
        link_vars_dict = {}

        link_on, link_off = self.link_encoder_.transform([True, False])

        # account for compat features
        if self.compat_features:
            assert compat_potentials.shape[0] == n_links
            compats = compat_potentials
        else:
            compats = (compat_potentials for _ in range(n_links))

        for (src, trg), link_v, compat in zip(x.link_to_prop,
                                              link_vars,
                                              compats):
            src_v = prop_vars[src]
            trg_v = prop_vars[trg]
            compat_factors.append(g.create_factor_dense([src_v, trg_v, link_v],
                                                        compat.ravel()))

            # keep track of binary link variables, for constraints.
            # we need .get_state() to get the underlaying PBinaryVariable
            link_vars_dict[src, trg] = link_v.get_state(link_on)

        # second-order factors
        coparent_factors = []
        grandparent_factors = []
        sibling_factors = []

        for score, (a, b, c) in zip(coparent_potentials, x.second_order):
            # a -> b <- c
            vars = [link_vars_dict[a, b], link_vars_dict[c, b]]
            coparent_factors.append(g.create_factor_pair(vars, score))

        for score, (a, b, c) in zip(grandparent_potentials, x.second_order):
            # a -> b -> c
            vars = [link_vars_dict[a, b], link_vars_dict[b, c]]
            grandparent_factors.append(g.create_factor_pair(vars, score))

        for score, (a, b, c) in zip(sibling_potentials, x.second_order):
            # a <- b -> c
            vars = [link_vars_dict[b, a], link_vars_dict[b, c]]
            sibling_factors.append(g.create_factor_pair(vars, score))

        # domain-specific constraints
        if constraints and 'cdcp' in constraints:

            # antisymmetry: if a -> b, then b cannot -> a
            for src in range(n_props):
                for trg in range(src):
                    fwd_link_v = link_vars_dict[src, trg]
                    rev_link_v = link_vars_dict[trg, src]
                    g.create_factor_logic('ATMOSTONE',
                                          [fwd_link_v, rev_link_v],
                                          [False, False])

            # transitivity.
            # forall a != b != c: a->b and b->c imply a->c
            for a, b, c in permutations(range(n_props), 3):
                ab_link_v = link_vars_dict[a, b]
                bc_link_v = link_vars_dict[b, c]
                ac_link_v = link_vars_dict[a, c]
                g.create_factor_logic('IMPLY',
                                      [ab_link_v, bc_link_v, ac_link_v],
                                      [False, False, False])

            # standard model:
            if 'strict' in constraints:
                for src, trg in x.link_to_prop:
                    src_v = prop_vars[src]
                    trg_v = prop_vars[trg]

                    for types in CDCP_ILLEGAL_LINKS:
                        src_ix, trg_ix = self.prop_encoder_.transform(types)
                        g.create_factor_logic('IMPLY',
                                              [src_v.get_state(src_ix),
                                               trg_v.get_state(trg_ix),
                                               link_vars_dict[src, trg]],
                                              [False, False, True])

        elif constraints and 'ukp' in constraints:

            # Tree constraints using AD3 MST factor for each paragraph.
            # First, identify paragraphs
            prop_para = np.array(x.prop_para)
            link_para = prop_para[x.link_to_prop[:, 0]]
            tree_factors = []

            for para_ix in np.unique(link_para):
                props = np.where(prop_para == para_ix)[0]
                offset = props.min()
                para_vars = []
                para_arcs = []  # call them arcs, semantics differ from links

                # add a new head node pointing to every possible variable
                for relative_ix, prop_ix in enumerate(props, 1):
                    para_vars.append(g.create_binary_variable())
                    para_arcs.append((0, relative_ix))

                # add an MST arc for each link
                for src, trg in x.link_to_prop[link_para == para_ix]:
                    relative_src = src - offset + 1
                    relative_trg = trg - offset + 1
                    para_vars.append(link_vars_dict[src, trg])
                    # MST arcs have opposite direction from argument links!
                    # because each prop can have multiple supports but not
                    # the other way around
                    para_arcs.append((relative_trg, relative_src))

                tree = fg.PFactorTree()
                g.declare_factor(tree, para_vars, True)
                tree.initialize(1 + len(props), para_arcs)
                tree_factors.append(tree)

            if 'strict' in constraints:
                # further domain-specific constraints
                mclaim_ix, claim_ix, premise_ix = self.prop_encoder_.transform(
                    ['MajorClaim', 'Claim', 'Premise'])

                # a -> b implies a = 'premise'
                for (src, trg), link_v in zip(x.link_to_prop, link_vars):
                    src_v = prop_vars[src]
                    g.create_factor_logic('IMPLY',
                                          [link_v.get_state(link_on),
                                           src_v.get_state(premise_ix)],
                                          [False, False])


        g.fix_multi_variables_without_factors()
        g.set_eta_ad3(eta)
        g.adapt_eta_ad3(adapt)
        g.set_max_iterations_ad3(max_iter)

        if exact:
            val, posteriors, additionals, status = g.solve_exact_map_ad3()
        else:
            val, posteriors, additionals, status = g.solve_lp_map_ad3()

        status = ["integer", "fractional", "infeasible", "not solved"][status]

        prop_marg = posteriors[:n_props * n_prop_classes]
        prop_marg = np.array(prop_marg).reshape(n_props, -1)

        link_marg = posteriors[n_props * n_prop_classes:]
        # remaining posteriors are for artificial root nodes for MST factors
        link_marg = link_marg[:n_links * n_link_classes]
        link_marg = np.array(link_marg).reshape(n_links, -1)

        n_compat = n_links * n_link_classes * n_prop_classes ** 2
        compat_marg = additionals[:n_compat]
        compat_marg = np.array(compat_marg).reshape((n_links,
                                                     n_prop_classes,
                                                     n_prop_classes,
                                                     n_link_classes))

        second_ordermarg = np.array(additionals[n_compat:])

        posteriors = (prop_marg, link_marg)
        additionals = (compat_marg, second_ordermarg)

        if relaxed:
            y_hat = posteriors, additionals
        else:
            y_hat = self._round(prop_marg, link_marg, prop_potentials,
                                link_potentials)

        if return_energy:
            return y_hat, status, -val
        else:
            return y_hat, status

    def _score(self, Y_true, Y_pred):

        acc = sum(1 for y_true, y_pred in zip(Y_true, Y_pred)
                  if np.all(y_true.links == y_pred.links) and
                  np.all(y_true.nodes == y_pred.nodes))
        acc /= len(Y_true)

        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore')
            link_macro, link_micro = arg_f1_scores(
                (y.links for y in Y_true),
                (y.links for y in Y_pred),
                average='binary',
                pos_label=True,
                labels=self.link_encoder_.classes_
            )

            node_macro, node_micro = arg_f1_scores(
                (y.nodes for y in Y_true),
                (y.nodes for y in Y_pred),
                average='macro',
                labels=self.prop_encoder_.classes_
            )

        return link_macro, link_micro, node_macro, node_micro, acc


class ArgumentGraphCRF(BaseArgumentMixin, StructuredModel):
    def __init__(self, class_weight=None, link_node_weight_ratio=1,
                 exact=False, constraints=None, compat_features=False,
                 coparents=False, grandparents=False, siblings=False):
        self.class_weight = class_weight
        self.link_node_weight_ratio = link_node_weight_ratio
        self.exact = exact
        self.constraints = constraints
        self.compat_features = compat_features
        self.coparents = coparents
        self.grandparents = grandparents
        self.siblings = siblings

        self.n_second_order_factors_ = coparents + grandparents + siblings
        self.n_prop_states = None
        self.n_link_states = None
        self.n_prop_features = None
        self.n_link_features = None

        self.n_second_order_features_ = None
        self.n_compat_features_ = None
        self.inference_calls = 0

        super(ArgumentGraphCRF, self).__init__()

    def initialize(self, X, Y):
        # each x in X is a vectorized doc exposing sp.csr x.X_prop, x.X_link,
        # and maybe x.X_compat and x.X_sec_ord
        # each y in Y exposes lists y.nodes, y.links

        x = X[0]
        self.n_prop_features = x.X_prop.shape[1]
        self.n_link_features = x.X_link.shape[1]

        if self.compat_features:
            self.n_compat_features_ = x.X_compat.shape[1]

        if self.n_second_order_factors_:
            self.n_second_order_features_ = x.X_sec_ord.shape[1]
        else:
            self.n_second_order_features_ = 0

        self.initialize_labels(Y)
        self._set_size_joint_feature()

        self.coparents_ = self.coparents
        self.grandparents_ = self.grandparents
        self.siblings_ = self.siblings

    def _set_size_joint_feature(self):  # assumes no second order
        compat_size = self.n_prop_states ** 2 * self.n_link_states
        if self.compat_features:
            compat_size  *= self.n_compat_features_

        total_n_second_order = (self.n_second_order_features_ *
                                self.n_second_order_factors_)

        self.size_joint_feature = (self.n_prop_features * self.n_prop_states +
                                   self.n_link_features * self.n_link_states +
                                   compat_size + total_n_second_order)

        logging.info("Joint feature size: {}".format(self.size_joint_feature))

    def joint_feature(self, x, y):

        if isinstance(y, DocLabel):
            Y_prop, Y_link, compat, second_order = self._marg_rounded(x, y)
        else:
            Y_prop, Y_link, compat, second_order = self._marg_fractional(x, y)

        prop_acc = safe_sparse_dot(Y_prop.T, x.X_prop)  # node_cls * node_feats
        link_acc = safe_sparse_dot(Y_link.T, x.X_link)  # link_cls * link_feats

        f_sec_ord = []

        if len(second_order):
            second_order = second_order.reshape(-1, len(x.second_order))
            if self.coparents:
                f_sec_ord.append(safe_sparse_dot(second_order[0], x.X_sec_ord))
                second_order = second_order[1:]

            if self.grandparents:
                f_sec_ord.append(safe_sparse_dot(second_order[0], x.X_sec_ord))
                second_order = second_order[1:]

            if self.siblings:
                f_sec_ord.append(safe_sparse_dot(second_order[0], x.X_sec_ord))

        elif self.n_second_order_factors_:
            # document has no second order factors so the joint feature
            # must be filled with zeros manually
            f_sec_ord = [np.zeros(self.n_second_order_features_)
                         for _ in range(self.n_second_order_factors_)]

        jf = np.concatenate([prop_acc.ravel(), link_acc.ravel(),
                             compat.ravel()] + f_sec_ord)

        return jf

    # basically reversing the joint feature
    def _get_potentials(self, x, w):
        # check sizes?
        n_node_coefs = self.n_prop_states * self.n_prop_features
        n_link_coefs = self.n_link_states * self.n_link_features
        n_compat_coefs = self.n_prop_states ** 2 * self.n_link_states

        if self.compat_features:
            n_compat_coefs *= self.n_compat_features_

        assert w.size == (n_node_coefs + n_link_coefs + n_compat_coefs +
                          self.n_second_order_features_ *
                          self.n_second_order_factors_)

        w_node = w[:n_node_coefs]

        w_node = w_node.reshape(self.n_prop_states, self.n_prop_features)

        w_link = w[n_node_coefs:n_node_coefs + n_link_coefs]
        w_link = w_link.reshape(self.n_link_states, self.n_link_features)

        # for readability, consume w. This is not inplace, don't worry.
        w = w[n_node_coefs + n_link_coefs:]
        w_compat = w[:n_compat_coefs]

        if self.compat_features:
            w_compat = w_compat.reshape((self.n_compat_features_, -1))
            w_compat = np.dot(x.X_compat, w_compat)
            compat_potentials = w_compat.reshape((-1,
                                                  self.n_prop_states,
                                                  self.n_prop_states,
                                                  self.n_link_states))
        else:
            compat_potentials = w_compat.reshape(self.n_prop_states,
                                                 self.n_prop_states,
                                                 self.n_link_states)

        w = w[n_compat_coefs:]

        coparent_potentials = grandparent_potentials = sibling_potentials = []

        if self.coparents:
            w_coparent = w[:self.n_second_order_features_]
            coparent_potentials = safe_sparse_dot(x.X_sec_ord, w_coparent)
            w = w[self.n_second_order_features_:]

        if self.grandparents:
            w_grandparent = w[:self.n_second_order_features_]
            grandparent_potentials = safe_sparse_dot(x.X_sec_ord,
                                                     w_grandparent)
            w = w[self.n_second_order_features_:]

        if self.siblings:
            w_sibling = w[:self.n_second_order_features_]
            sibling_potentials = safe_sparse_dot(x.X_sec_ord, w_sibling)

        prop_potentials = safe_sparse_dot(x.X_prop, w_node.T)
        link_potentials = safe_sparse_dot(x.X_link, w_link.T)

        return (prop_potentials, link_potentials, compat_potentials,
                coparent_potentials, grandparent_potentials,
                sibling_potentials)

    def inference(self, x, w, relaxed=False, return_energy=False):

        self.inference_calls += 1

        potentials = self._get_potentials(x, w)
        out = self._inference(x, potentials, exact=self.exact,
                              relaxed=relaxed, return_energy=return_energy,
                              constraints=self.constraints)
        if return_energy:
            return out[0], out[-1]
        else:
            return out[0]

    def loss_augmented_inference(self, x, y, w, relaxed=None):

        self.inference_calls += 1

        potentials = self._get_potentials(x, w)

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

        potentials = (prop_potentials,
                      link_potentials,
                      compat_potentials,
                      coparent_potentials,
                      grandparent_potentials,
                      sibling_potentials)

        out = self._inference(x, potentials, exact=self.exact,
                              relaxed=relaxed, constraints=self.constraints)
        return out[0]
