"""Assuming test predictions are available, compute and display scores."""


import os
import sys
import warnings

import numpy as np
import dill

from sklearn.metrics import precision_recall_fscore_support, f1_score

from marseille.datasets import get_dataset_loader
from marseille.custom_logging import logging


def arg_p_r_f(Y_true, Y_pred, labels, **kwargs):

    macro_p = []
    macro_r = []
    macro_f = []

    micro_true = []
    micro_pred = []

    for y_true, y_pred in zip(Y_true, Y_pred):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     **kwargs)
        macro_p.append(p)
        macro_r.append(r)
        macro_f.append(f)

        micro_true.extend(y_true)
        micro_pred.extend(y_pred)

    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        micro_true, micro_pred, **kwargs
    )
    kwargs.pop('average')
    per_class_fs = f1_score(micro_true, micro_pred, average=None, **kwargs)

    res = {
        'p_macro': np.mean(macro_p),
        'r_macro': np.mean(macro_r),
        'f_macro': np.mean(macro_f),
        'p_micro': micro_p,
        'r_micro': micro_r,
        'f_micro': micro_f
    }

    for label, per_class_f in zip(sorted(labels), per_class_fs):
        res['f_class_{}'.format(label)] = per_class_f

    return res


def compute_scores(Y_true, Y_pred, prop_labels, link_labels):

    # hard accuracy
    acc = sum(1 for y_true, y_pred in zip(Y_true, Y_pred)
              if np.all(y_true.links == y_pred.links) and
              np.all(y_true.nodes == y_pred.nodes))
    acc /= len(Y_true)

    with warnings.catch_warnings() as w:
        warnings.simplefilter('ignore')
        link_results = arg_p_r_f(
            (y.links for y in Y_true),
            (y.links for y in Y_pred),
            labels=link_labels,
            average='binary',
            pos_label=True
        )
        prop_results = arg_p_r_f(
            (y.nodes for y in Y_true),
            (y.nodes for y in Y_pred),
            labels=prop_labels,
            average='macro',
        )
    scores = {"prop_{}".format(key): val for key, val in prop_results.items()}
    scores.update({"link_{}".format(key): val for key, val in
                  link_results.items()})
    scores['avg_f_micro'] =  0.5 * (scores['link_f_micro'] +
                                    scores['prop_f_micro'])
    scores['accuracy'] = acc

    return scores


# tpl = os.path.join("test_results", "{}_{}_{}.predictions.dill")
tpl = os.path.join("test_results",
                   # "exact_predictions",
                   "exact=True_{}_{}_{}.predictions.dill")

if __name__ == '__main__':
    dataset = sys.argv[1]

    if dataset not in ('cdcp', 'ukp'):
        raise ValueError("Unknown dataset {}. "
                         "Supported: ukp|cdcp.".format(dataset))

    link_labels = [False, True]
    prop_labels = (['MajorClaim', 'Claim', 'Premise'] if dataset == 'ukp'
                   else ['value', 'policy', 'testimony', 'fact', 'reference'])

    # get true test labels
    load_te, ids_te = get_dataset_loader(dataset, split='test')
    Y_true = [doc.label for doc in load_te(ids_te)]

    print("dataset={}".format(dataset))

    scores = dict()
    for method in ("linear", "linear-struct", "rnn", "rnn-struct"):
        scores[method] = dict()
        for model in ("bare", "full", "strict"):
            scores_ = scores[method][model] = dict()

            fn = tpl.format(dataset, method, model)

            if not os.path.isfile(fn):
                logging.info("Could not find {}".format(fn))
                continue

            with open(fn, "rb") as f:
                Y_pred = dill.load(f)

            # compute test scores:
            scores[method][model] = compute_scores(Y_true,
                                                   Y_pred,
                                                   prop_labels,
                                                   link_labels)

    pretty = {'avg_f_micro': 'Average $F_1$',
              'accuracy': 'Accuracy',
              'link_f_micro': '{\Link} $F_1$',
              'link_p_micro': '{\Link} $P$',
              'link_r_micro': '{\Link} $R$',
              'prop_f_micro': '{\Prop} $F_1$',
              'prop_p_micro': '{\Prop} $P$',
              'prop_r_micro': '{\Prop} $R$',
              'prop_f_class_MajorClaim': 'MajorClaim $F_1$',
              'prop_f_class_Claim': 'Claim $F_1$',
              'prop_f_class_Premise': 'Premise $F_1$',
              'prop_f_class_fact': 'Fact $F_1$',
              'prop_f_class_value': 'Value $F_1$',
              'prop_f_class_policy': 'Policy $F_1$',
              'prop_f_class_testimony': 'Testimony $F_1$',
              'prop_f_class_reference': 'Reference $F_1$'}

    pretty = {'avg_f_micro': 'Average',
              'link_f_micro': '{\Link}',
              'prop_f_micro': '{\Prop}',
              'prop_f_class_MajorClaim': '{\quad}MajorClaim',
              'prop_f_class_Claim': '{\quad}Claim',
              'prop_f_class_Premise': '{\quad}Premise',
              'prop_f_class_fact': '{\quad}Fact',
              'prop_f_class_value': '{\quad}Value',
              'prop_f_class_policy': '{\quad}Policy',
              'prop_f_class_testimony': '{\quad}Testimony',
              'prop_f_class_reference': '{\quad}Reference'}

    # keys = ['avg_f_micro', 'link_f_micro', 'link_p_micro', 'link_r_micro',
    #         'prop_f_micro', 'prop_p_micro', 'prop_r_micro']
    # keys += ['prop_f_class_{}'.format(lbl) for lbl in prop_labels]
    # keys += ['accuracy']
    keys = ['avg_f_micro', 'link_f_micro', 'prop_f_micro']
    keys += ['prop_f_class_{}'.format(lbl) for lbl in prop_labels]

    def _row(numbers):
        argmax = np.argmax(numbers)
        strs = ["{:.1f} ".format(100 * x) for x in numbers]
        strs[argmax] = "{\\bf %s}" % strs[argmax][:-1]
        strs = [s.rjust(10) for s in strs]
        return " & ".join(strs)

    # keys = ['avg_f_micro', 'link_f_micro', 'link_p_micro', 'link_r_micro',
    #         'prop_f_micro', 'prop_p_micro', 'prop_r_micro']
    # keys += ['prop_f_class_{}'.format(lbl) for lbl in prop_labels]
    # keys += ['accuracy']
    keys = ['avg_f_micro', 'link_f_micro', 'prop_f_micro']
    keys += ['prop_f_class_{}'.format(lbl) for lbl in prop_labels]

    def _row(numbers):
        argmax = np.argmax(numbers)
        strs = ["{:.1f} ".format(100 * x) for x in numbers]
        strs[argmax] = "{\\bf %s}" % strs[argmax][:-1]
        strs = [s.rjust(10) for s in strs]
        return " & ".join(strs)

    for key in keys:
        print("{:>20}".format(pretty[key]), "&", _row([
                scores[method][model].get(key, -1)
                for method in ('linear', 'rnn', 'linear-struct', 'rnn-struct')
                for model in ('bare', 'full', 'strict')]),
            r"\\")


