# marseille

**m**ining **ar**gument **s**tructures with **e**xpressive **i**nference (with **l**inear and **l**stm **e**ngines)


## What is it?

Marseille learns to predict argumentative proposition types and the support
relations between them, as inference in a expressive factor graph.

Read more about it in [our paper](https://arxiv.org/abs/1704.06869),

> Vlad Niculae, Joonsuk Park, Claire Cardie.
> Argument Mining with Structured SVMs and RNNs.
> In: Proceedings of ACL, 2017.

If you find this project useful, you may cite us using:

```
@inproceedings{niculae17marseille,
  author={Vlad Niculae and Joonsuk Park and Claire Cardie},
  title={{Argument Mining with Structured SVMs and RNNs}},
  booktitle={Proceedings of ACL},
  year=2017
}
```

## Requirements

 - numpy
 - scipy
 - scikit-learn
 - pystruct
 - nltk
 - dill
 - docopt
 - [dynet](https://github.com/clab/dynet)
 - [lightning](https://github.com/scikit-learn-contrib/lightning)
 - [ad3 (development version)](https://github.com/vene/ad3/tree/newrel)


## Usage

(replace `$ds` with `cdcp` or `ukp`)

0. download the data from http://joonsuk.org/ and unzip it in the subdirectory `data`, i.e. the path
`./data/process/erule/train/` is valid.

1. extract relevant subset of GloVe embeddings:
```
    python -m marseille.preprocess embeddings $ds --glove-file=/p/glove.840B.300d.txt
```

2. extract features:
```
    python -m marseille.features $ds

    # (for cdcp only:)
    python -m marseille.features cdcp-test
```

3. generate vectorized train-test split (for baselines only)
```
    mkdir data/process/.../
    python -m marseille.vectorize split cdcp
```

4. run chosen model, for example:
```
    python -m experiments.exp_train_test $ds --method rnn-struct --model strict
```
(for dynet models, set `--dynet-seed=42` for exact reproducibility)

5. compare results:
```
    python -m experiments.plot_test_results.py $ds
```

To reproduce cross-validation model selection, you also would need to run:

```
    python -m marseille.vectorize folds $ds
```


#  Running a model on your own data:

If you have some documents e.g. F.txt, G.txt that you would like to run a
pretrained model on, read on.

1. download the required preprocessing toolkits:
   [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (tested
   with version 3.6.0) and the
   [WING-NUS PDTB discourse parser](https://github.com/WING-NUS/pdtb-parser)
   (tested with
   [this commit](https://github.com/WING-NUS/pdtb-parser/commit/5ee603a9))
   and configure their paths:

```
    export MARSEILLE_CORENLP_PATH=/home/vlad/corenlp  #  path to CoreNLP
    export MARSEILLE_WINGNUS_PATH=/home/vlad/wingnus  #  path to WING-NUS parser
```

  Note: If you already generated F.txt.json with CoreNLP and F.txt.pipe with the
  WING-NUS parser (e.g., on a different computer), you may skip this step and
  *marseille* will detect those files automatically.

  Otherwise, these files are generated the first time that a `UserDoc` object
  is instantiated for a given document. In particular, the step below will do
  this automatically.

2. extract the features:

```
    python -m marseille.features user F G  # raw input must be in F.txt & G.txt
```

  This is needed for the RNN models too, because the feature files encode some
  metadata about the document structure.

3. predict, e.g. using the model saved in step 4 above:

```
    python -m experiments.predict_pretrained --method=rnn-struct \
    test_results/exact=True_cdcp_rnn-struct_strict F G
```

