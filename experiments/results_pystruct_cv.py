import numpy as np
import sys

from marseille.io import load_results
from marseille.custom_logging import logging

if __name__ == '__main__':
    dataset = sys.argv[1]


    pystruct_df = []
    class_weight = 'balanced'

    Cs = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
    constraint_vals = ('', dataset, dataset + '+strict')

    for C in Cs:
        for constraints in constraint_vals:
            for extras in (False, True):
                compat_features = extras
                second_order_features = extras
                try:
                    results = load_results("svmstruct_cv_score",
                                           (dataset, C, class_weight,
                                            constraints,
                                            compat_features,
                                            second_order_features))
                except Exception as e:
                    logging.info("not loaded: dataset={} C={} cw={} "
                                 "constraints={} compat_features={} "
                                 "second_order_features={} {}".format(
                        dataset, C, class_weight, constraints,
                        compat_features, second_order_features, e
                    ))
                    continue
                scores, _ = results
                scores = np.mean(scores, axis=0)

                link_macro, link_micro, node_macro, node_micro, acc = scores
                pystruct_df.append(dict(C=C,
                                        constraints=constraints,
                                        second_order=second_order_features,
                                        compat_features=compat_features,
                                        link_macro=link_macro,
                                        link_micro=link_micro,
                                        node_macro=node_macro,
                                        node_micro=node_micro,
                                        accuracy=acc))


    print(len(pystruct_df))
    import json
    with open("svm_cv_{}.json".format(sys.argv[1]), "w") as f:
        json.dump(pystruct_df, f)
