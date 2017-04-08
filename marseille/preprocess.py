"""Run preprocessing on the data."""

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

import os
import json

import numpy as np

from marseille.custom_logging import logging


def write_files(file, base_dir):
    for doc in file:
        with open(os.path.join(base_dir, "{:05d}.txt".format(doc.doc_id)),
                  "w") as f:
            f.write(doc.text)

        with open(os.path.join(base_dir, "{:05d}.ann.json".format(doc.doc_id)),
                  "w") as f:
            metadata = {'prop_offsets': doc.prop_offsets,
                        'prop_labels': doc.prop_labels,
                        'reasons': doc.reasons,
                        'evidences': doc.evidences,
                        'url': doc.url}
            json.dump(metadata, f, sort_keys=True)


def _transitive(links):
    """perform transitive closure of links.

    For input [(1, 2), (2, 3)] the output is [(1, 2), (2, 3), (1, 3)]
    """

    old_links = links
    links = set(links)
    while True:
        new_links = [(src_a, trg_b)
                     for src_a, trg_a in links
                     for src_b, trg_b in links
                     if trg_a == src_b
                     and (src_a, trg_b) not in links]
        if new_links:
            links.update(new_links)
        else:
            break

    return links


def merge_prop_labels(labels):
    """After joining multiple propositions, we need to decide the new type.

    Rules:
        1. if the span is a single prop, keep the label
        2. if the span props have the same type, use that type
        3. Else, rules from Jon: policy>value>testimony>reference>fact
    """

    if len(labels) == 1:
        return labels[0]

    labels = set(labels)

    if len(labels) == 1:
        return next(iter(labels))

    if 'policy' in labels:
        return 'policy'
    elif 'value' in labels:
        return 'value'
    elif 'testimony' in labels:
        return 'testimony'
    elif 'reference' in labels:
        return 'reference'
    elif 'fact' in labels:
        return 'fact'
    else:
        raise ValueError("weird labels: {}".format(" ".join(labels)))


def merge_spans(doc, include_nonarg=True):
    """Normalization needed for CDCP data because of multi-prop spans"""

    # flatten multi-prop src spans like (3, 6) into new propositions
    # as long as they never overlap with other links. This inevitably will
    # drop some data but it's a very small number.

    # function fails if called twice because
    #    precondition: doc.links = [((i, j), k)...]
    #    postcondition: doc.links = [(i, k)...]

    new_links = []
    new_props = {}
    new_prop_offsets = {}

    dropped = 0

    for (start, end), trg in doc.links:

        if start == end:
            new_props[start] = (start, end)
            new_prop_offsets[start] = doc.prop_offsets[start]

            new_props[trg] = (trg, trg)
            new_prop_offsets[trg] = doc.prop_offsets[trg]

            new_links.append((start, trg))

        elif start < end:
            # multi-prop span. Check for problems:

            problems = []
            for (other_start, other_end), other_trg in doc.links:
                if start == other_start and end == other_end:
                    continue

                # another link coming out of a subset of our span
                if start <= other_start <= other_end <= end:
                    problems.append(((other_start, other_end), other_trg))

                # another link coming into a subset of our span
                if start <= other_trg <= end:
                    problems.append(((other_start, other_end), other_trg))

            if not len(problems):
                if start in new_props:
                    assert (start, end) == new_props[start]

                new_props[start] = (start, end)
                new_prop_offsets[start] = (doc.prop_offsets[start][0],
                                           doc.prop_offsets[end][1])

                new_props[trg] = (trg, trg)
                new_prop_offsets[trg] = doc.prop_offsets[trg]

                new_links.append((start, trg))

            else:
                # Since we drop the possibly NEW span, there is no need
                # to remove any negative links.
                dropped += 1

    if include_nonarg:
        used_props = set(k for a, b in new_props.values()
                         for k in range(a, b + 1))
        for k in range(len(doc.prop_offsets)):
            if k not in used_props:
                new_props[k] = (k, k)
                new_prop_offsets[k] = doc.prop_offsets[k]

    mapping = {key: k for k, key in enumerate(sorted(new_props))}
    doc.props = [val for _, val in sorted(new_props.items())]
    doc.prop_offsets = [val for _, val in sorted(new_prop_offsets.items())]
    doc.links = [(mapping[src], mapping[trg]) for src, trg in new_links]

    doc.prop_labels = [merge_prop_labels(doc.prop_labels[a:1 + b])
                       for a, b in doc.props]

    return doc


def test_merge_spans():
    from collections import Counter
    from marseille.datasets import get_dataset_loader

    load, ids = get_dataset_loader("cdcp", "train")
    n_nones = 0
    label_counts = Counter()
    for doc in load(ids):
        label_counts.update(doc.prop_labels)
        # drops 14 links in training and 8 in test split
        n_nones += sum(1 for x in doc.prop_labels if x is None)

    print(label_counts.most_common())
    print(n_nones)


def from_json():
    data_raw = "data/raw/erule/"
    data_process = "data/process/erule/"

    from marseille.argdoc import CdcpArgumentationDoc

    with open(os.path.join(data_raw, "cdcp.train.jsonlist")) as f:
        f = map(json.loads, f)
        f = map(CdcpArgumentationDoc.from_json, f)
        write_files(f, base_dir=os.path.join(data_process, 'train'))

    with open(os.path.join(data_raw, "cdcp.test.jsonlist")) as f:
        f = map(json.loads, f)
        f = map(CdcpArgumentationDoc.from_json, f)
        write_files(f, base_dir=os.path.join(data_process, 'test'))


def optimize_glove(glove_path, vocab):
    """Trim down GloVe embeddings to use only words in the data."""
    vocab_set = frozenset(vocab)
    seen_vocab = []
    X = []
    with open(glove_path) as f:
        for line in f:
            line = line.strip().split(' ')  # split() fails on ". . ."
            word, embed = line[0], line[1:]
            if word in vocab_set:
                X.append(np.array(embed, dtype=np.float32))
                seen_vocab.append(word)
    return seen_vocab, np.row_stack(X)


def store_optimized_embeddings(dataset, glove_path):

    from marseille.datasets import get_dataset_loader

    out_path = os.path.join('data', '{}-glove.npz'.format(dataset))
    vocab = set()
    load, ids = get_dataset_loader(dataset, "train")
    for doc in load(ids):
        vocab.update(doc.tokens())
    res = optimize_glove(glove_path, vocab)
    glove_vocab, glove_embeds = res
    coverage = len(glove_vocab) / len(vocab)
    np.savez(out_path, vocab=glove_vocab, embeds=glove_embeds)
    logging.info("GloVe coverage: {:.2f}%".format(100 * coverage))


if __name__ == '__main__':
    from docopt import docopt

    usage = """
        Usage:
            preprocess embeddings (cdcp|ukp) [--glove-file=F]
            preprocess from-json

        Options:
            --glove-file=F     Path to GloVe [default: glove.840B.300d.txt]
    """
    args = docopt(usage)

    if args['embeddings']:
        dataset = 'cdcp' if args['cdcp'] else 'ukp'
        store_optimized_embeddings(dataset, args['--glove-file'])

    elif args['from-json']:
        raise NotImplementedError()
    #     from_json()

