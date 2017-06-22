import pickle
from marseille.user_doc import UserDoc

if __name__ == '__main__':
    from docopt import docopt

    usage = """
    Usage:
        predict_pretrained --method=M [--exact --dynet-seed=N --dynet-mem=N] \
<model_root> <test_file>...

    Options:
        model_root: the root filename (no extension) of the pretrained model.
        test_file: the root name (no extension) of the .txt file(s) to predict.
        --method: the type of classifier we are loading. one of (linear,
            linear-struct, rnn, rnn-struct)
    """

    args = docopt(usage)

    model = args['<model_root>']
    test_files = args['<test_file>']

    test_docs = [UserDoc(test_file) for test_file in test_files]

    if args['--method'] in ('rnn', 'rnn-struct'):

        # load the pickled classifier husk
        with open('{}.model.pickle'.format(model), "rb") as fp:
            rnn = pickle.load(fp)

        # fill in the dynet parameter values
        rnn.load('{}.model.dynet'.format(model))

        Y_pred = rnn.predict(test_docs, exact=args['--exact'])

    elif args['--method'] == 'linear-struct':

        # load the vectorizers
        with open('{}.vectorizers.pickle'.format(model), "rb") as fp:
            vects = pickle.load(fp)

        # vectorize the test docs
        from .exp_svmstruct import _vectorize
        X = [_vectorize(doc, *vects) for doc in test_docs]

        # load the pickled classifier
        with open('{}.model.pickle'.format(model), "rb") as fp:
            clf = pickle.load(fp)

        clf.model.exact = args['--exact']
        Y_pred = clf.predict(X)

    else:
        raise NotImplementedError("Loading models of type {} not yet "
            "supported.".format(args['--method']))

    print(Y_pred)

