# marseille

**m**ining **ar**gument **s**tructures with **e**xpressive **i**nference (with **l**inear and **l**stm **e**ngines)


## What is it?

Marseille learns to predict argumentative proposition types and the support
relations between them, as inference in a expressive factor graph.

Read more about it in our paper,

> Argument mining with structured SVMs and RNNs.
> Vlad Niculae, Joonsuk Park, Claire Cardie. In: Proc. of ACL, 2017.
> (preprint coming soon)
    
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

(replace $ds with cdcp or ukp)

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

