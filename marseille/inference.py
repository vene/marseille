# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

import warnings

try:
    from pystruct.models.utils import loss_augment_unaries
except ImportError:
    def loss_augment_unaries(unary_potentials, y, class_weight):
        warnings.warn("PyStruct not installed, slow loss_augment_unaries.")
        n_states = unary_potentials.shape[1]
        for i in range(unary_potentials.shape[0]):
            for s in range(n_states):
                if s == y[i]:
                    continue
                unary_potentials[i, s] += class_weight[y[i]]

# "standard model"
# auto-generate:
# def _get_illegal_links():
#
#     illegal_links = []
#     order = ["reference", "testimony", "fact", "value", "policy"]
#
#     for trg in range(len(order)):
#         for src in range(trg + 1, len(order)):
#             illegal_links.append([order[src], order[trg]])
#
#     print(illegal_links)
#     return illegal_links

# CDCP_ILLEGAL_LINKS = _get_illegal_links()

CDCP_ILLEGAL_LINKS = [('testimony', 'reference'),
                      ('fact', 'reference'),
                      ('value', 'reference'),
                      ('policy', 'reference'),
                      ('fact', 'testimony'),
                      ('value', 'testimony'),
                      ('policy', 'testimony'),
                      ('value', 'fact'),
                      ('policy', 'fact'),
                      ('policy', 'value')]
