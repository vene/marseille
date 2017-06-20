"""Final experiments, trained on the full training set.

This uses hyperparameters chosen by cross-validation from exp_* """

import os
import pickle

import numpy as np

from marseille.user_doc import UserDoc
from marseille.datasets import get_dataset_loader, load_embeds
from marseille.custom_logging import logging
from marseille.argrnn import BaselineArgumentLSTM, ArgumentLSTM
from marseille.io import load_csr

from .exp_svmstruct import fit_predict as fit_pred_pystruct
from .exp_linear import BaselineStruct


if __name__ == '__main__':
    exact_test = True
    dataset = 'cdcp'

    load_tr, ids_tr = get_dataset_loader(dataset, split="train")
    train_docs = list(load_tr(ids_tr))[:20]

    filename = "pickle_test"

    constraints = ''
    compat_features = False
    second_order = False

    grandparents = coparents = siblings = False

    Y_train = [doc.label for doc in train_docs]

    pkl = True

    if pkl:
        embeds = load_embeds(dataset)
        rnn = ArgumentLSTM(lstm_dropout=0,
                           prop_mlp_layers=2,
                           score_at_iter=None,
                           n_mlp=128,
                           n_lstm=128,
                           lstm_layers=2,
                           link_mlp_layers=1,
                           embeds=embeds,
                           link_bilinear=True,
                           class_weight='balanced',
                           second_order_multilinear=True,
                           exact_inference=False,
                           constraints=constraints,
                           compat_features=compat_features,
                           grandparent_layers=(1 if grandparents else 0),
                           coparent_layers=(1 if coparents else 0),
                           sibling_layers=(1 if siblings else 0),
                           exact_test=exact_test,
                           max_iter=5, mlp_dropout=0.15)

        rnn.fit(train_docs, Y_train)

        with open('{}.model.pickle'.format(filename), "wb") as fp:
            pickle.dump(rnn, fp)

        rnn.save('{}.model.dynet'.format(filename))

    else:
        with open('{}.model.pickle'.format(filename), "rb") as fp:
            rnn = pickle.load(fp)

        rnn.load('{}.model.dynet'.format(filename))

    test_docs = [UserDoc("test")]
    Y_pred = rnn.predict(test_docs)
    print(Y_pred[0].nodes, Y_pred[0].links)
