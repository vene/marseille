"""For each model, find the test documents where it
    (1) wins by the largest margin
    (2) loses by the largest margin
"""

import os
import sys
import warnings
import subprocess

import dill
import numpy as np

from sklearn.metrics import f1_score

from marseille.datasets import get_dataset_loader
from marseille.custom_logging import logging

# tpl = os.path.join("test_results", "{}_{}_{}.predictions.dill")

# FROM = 'second' # median
# FROM = 'median'
FROM = 'other'


tpl = os.path.join("test_results",
                   "exact_predictions",
                   "exact=True_{}_{}_{}.predictions.dill")

def scores_per_doc(Y_true, Y_pred, prop_labels):
    macro_f_prop = [f1_score(y_true.nodes, y_pred.nodes,
                            labels=prop_labels,
                            average='macro')
                    for y_true, y_pred in zip(Y_true, Y_pred)]

    macro_f_link = [f1_score(y_true.links, y_pred.links,
                             labels=[False, True],
                             average='binary',
                             pos_label=True)
                    for y_true, y_pred in zip(Y_true, Y_pred)]

    # return 0.5 * (np.array(macro_f_prop) + np.array(macro_f_link))
    return macro_f_link


def margins(doc_scores):
    margin_win = np.zeros_like(doc_scores)
    margin_lose = np.zeros_like(doc_scores)

    for j in range(doc_scores.shape[1]):
        my_scores = doc_scores[:, j]
        others = np.delete(doc_scores, j, axis=1)

        if FROM  == 'second':
            margin_win[:, j] = np.maximum(my_scores - others.max(axis=1), 0)
            margin_lose[:, j] = np.maximum(others.min(axis=1) - my_scores, 0)
        if FROM  == 'other':
            margin_win[:, j] = np.maximum(my_scores - others.min(axis=1), 0)
            margin_lose[:, j] = np.maximum(others.max(axis=1) - my_scores, 0)
        elif FROM == 'median':
            margin_win[:, j] = np.maximum(my_scores - np.median(others,
                                          axis=1), 0)
            margin_lose[:, j] = np.maximum(np.median(others, axis=1) -
                                           my_scores, 0)


    return margin_win, margin_lose


def render_doc(doc):
    ret = '<h4>id={}</h4><ol>'.format(doc.doc_id)
    for start, end in doc.prop_offsets:
        ret += '<li>{}</li>'.format(doc.text[start:end])
    ret += '</ol>'
    return ret


def _svg(labels, links):
    head = """\
    digraph G{
    edge [dir=forward]
    node [shape=plaintext]"""

    lbl_code = ['{} [label="{}"]'.format(k, w)
                for k, w in enumerate(labels)]

    link_code = ['{} -> {}'.format(*link) for link in links]

    dot_string = "\n".join([head] + lbl_code + link_code + ["}"])

    try:
        process = subprocess.Popen(
            ['dot', '-Tsvg'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except OSError:
        raise Exception('Cannot find the dot binary from Graphviz package')
    out, err = process.communicate(dot_string)
    if err:
        raise Exception(
            'Cannot create svg representation by running dot from string: {}'
            ''.format(dot_string))

    return out


def render_prediction(doc, Y):
    labels = ['({}) {:.2}'.format(i, lbl) for i, lbl in enumerate(Y.nodes, 1)]
    links = doc.link_to_prop[Y.links]

    return _svg(labels, links)


if __name__ == '__main__':
    dataset = sys.argv[1]
    load, ids = get_dataset_loader(dataset, split="test")
    docs = list(load(ids))
    Y_true = [doc.label for doc in docs]

    prop_labels = (['MajorClaim', 'Claim', 'Premise'] if dataset == 'ukp'
                   else ['value', 'policy', 'testimony', 'fact', 'reference'])

    predictions = dict()
    model_names = []
    doc_scores = []
    for method in ("linear", "linear-struct", "rnn", "rnn-struct"):
        for model in ("bare", "full", "strict"):

            fn = tpl.format(dataset, method, model)

            if not os.path.isfile(fn):
                logging.info("Could not find {}".format(fn))
                continue

            with open(fn, "rb") as f:
                predictions[method, model] = Y_pred = dill.load(f)

            model_names.append((method, model))
            with warnings.catch_warnings() as w:
                warnings.simplefilter('ignore')
                doc_scores.append(scores_per_doc(Y_true, Y_pred, prop_labels))

    doc_scores = np.array(doc_scores).T  # n_samples x n_models
    margin_win, margin_lose = margins(doc_scores)

    for k, name in enumerate(model_names):
        fn = os.path.join("res", "error_analysis", "{}_{}_{}_{}.html".format(
            FROM, dataset, *name))
        model_margin_win = margin_win[:, k]
        model_margin_lose = margin_lose[:, k]
        n_wins = np.sum(model_margin_win > 0)
        n_losses = np.sum(model_margin_lose > 0)

        html = "<html><h1>{}</h1>".format(" ".join(name))
        html += "<h2>from={}<h2>".format(FROM)

        win_ix = np.argsort(model_margin_win)[::-1][:n_wins]
        lose_ix = np.argsort(model_margin_lose)[::-1][:n_losses]

        html += "<h2>Wins</h2>"

        for ix in win_ix:
            html += render_doc(docs[ix])
            html += "<h4>True Y</h4><div>{}</div>".format(
                render_prediction(docs[ix], Y_true[ix])
            )

            html += '<h4>{} ({:.3f})</h4>'.format(name, doc_scores[ix].max())
            html += render_prediction(docs[ix], predictions[name][ix])

            other = np.argsort(doc_scores[ix])[::-1][1]
            html += '<h4> second best prediction: {} ({})</h4>'.format(
                model_names[other], doc_scores[ix, other])
            html += render_prediction(docs[ix],
                                      predictions[model_names[other]][ix])

            other = np.argsort(doc_scores[ix])[::-1][-1]
            html += '<h4> worst prediction: {} ({})</h4>'.format(
                model_names[other], doc_scores[ix, other])
            html += render_prediction(docs[ix],
                                      predictions[model_names[other]][ix])

        html += "<hr />"
        html += "<h2>Losses</h2>"
        for ix in lose_ix:
            html += render_doc(docs[ix])
            html += "<h4>True Y</h4><div>{}</div>".format(
                render_prediction(docs[ix], Y_true[ix])
            )

            html += '<h4>{} ({:.3f})</h4>'.format(name, doc_scores[ix].min())
            html += render_prediction(docs[ix], predictions[name][ix])

            other = np.argsort(doc_scores[ix])[1]
            html += '<h4> second worst prediction: {} ({})</h4>'.format(
                model_names[other], doc_scores[ix, other])
            html += render_prediction(docs[ix],
                                      predictions[model_names[other]][ix])

            other = np.argsort(doc_scores[ix])[-1]
            html += '<h4> best prediction: {} ({})</h4>'.format(
                model_names[other], doc_scores[ix, other])
            html += render_prediction(docs[ix],
                                      predictions[model_names[other]][ix])

        html += '</html>'

        with open(fn, "w") as f:
            print(html, file=f)
