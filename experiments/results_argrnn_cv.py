import numpy as np
import sys
from marseille.custom_logging import logging
from marseille.io import load_results


def combine_scores(scores, score_at_iter):
    # patch up early-stopped scores
    for fold_scores in scores:
        while len(fold_scores) < len(score_at_iter):
            fold_scores.append(fold_scores[-1])

    return np.mean(scores, axis=0)


if __name__ == '__main__':

    # defaults
    dataset = sys.argv[1]
    dynet_weight_decay = None
    mlp_dropout = 0.1
    rnn_dropout = 0.0
    prop_layers = 2
    constraints = ""

    baseline_df = []
    full_df = []

    for mlp_dropout in (0.05, 0.10, 0.15, 0.20, 0.25):
        for constraints in ("", dataset, dataset + "+strict"):
            # get the baseline results
            try:
                results = load_results("baseline_argrnn_cv_score",
                                       (dataset, dynet_weight_decay, mlp_dropout,
                                        rnn_dropout, prop_layers, constraints))
            except Exception as e:
                logging.info("not loaded: dataset={} dynet_weight_decay={} "
                             "mlp_dropout={} rnn_dropout={} prop_layers={} "
                             "constraints={} error={}".format(
                    dataset, dynet_weight_decay, mlp_dropout,
                    rnn_dropout, prop_layers, constraints, e))
                continue

            scores, score_at_iter, _ = results
            avg_scores = combine_scores(scores, score_at_iter)

            for iter, score in zip(score_at_iter, avg_scores):
                link_macro, link_micro, node_macro, node_micro, acc = score
                baseline_df.append(dict(mlp_dropout=mlp_dropout,
                                        constraints=constraints,
                                        iter=iter,
                                        link_macro=link_macro,
                                        link_micro=link_micro,
                                        node_macro=node_macro,
                                        node_micro=node_micro,
                                        accuracy=acc))

    for mlp_dropout in (0.05, 0.10, 0.15, 0.20, 0.25):
        for constraints in ("", dataset, dataset + "+strict"):
            # get the full blown experiments
            balanced = 'balanced'
            for extra in (False, True):
                compat_features = extra
                second_order = extra
                try:
                    results = load_results("argrnn_cv_score",
                                           (dataset, dynet_weight_decay,
                                            mlp_dropout, rnn_dropout,
                                            prop_layers, balanced, constraints,
                                            compat_features, second_order))
                except Exception as e:
                    logging.info("not loaded: dataset={} dynet_weight_decay={} "
                                 "mlp_dropout={} rnn_dropout={} prop_layers={} "
                                 "cw={} constraints={} compat_features={} "
                                 "second_order={} error={}".format(
                        dataset, dynet_weight_decay, mlp_dropout,
                        rnn_dropout, prop_layers, balanced, constraints,
                        compat_features, second_order, e))
                    continue

                scores, score_at_iter, _ = results
                avg_scores = combine_scores(scores, score_at_iter)

                for iter, score in zip(score_at_iter, avg_scores):
                    link_macro, link_micro, node_macro, node_micro, acc = score
                    full_df.append(dict(mlp_dropout=mlp_dropout,
                                        constraints=constraints,
                                        balanced=balanced,
                                        compat_features=compat_features,
                                        second_order=second_order,
                                        iter=iter,
                                        link_macro=link_macro,
                                        link_micro=link_micro,
                                        node_macro=node_macro,
                                        node_micro=node_micro,
                                        accuracy=acc))


    print(len(baseline_df))
    print(len(full_df))
    import json
    with open("rnn_cv_{}.json".format(sys.argv[1]), "w") as f:
        json.dump([baseline_df, full_df], f)
