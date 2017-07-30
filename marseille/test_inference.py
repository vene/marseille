import pytest
import numpy as np
from marseille.argdoc import DocLabel
from marseille.struct_models import BaseArgumentMixin
from .test_argrnn import ArgDocStub

class InferenceStub(BaseArgumentMixin):
    compat_features = False
    class_weight = None
    def __init__(self, constraints):
        self.constraints = constraints
        if 'cdcp' in constraints:
            self.prop_types = ['fact', 'value', 'policy', 'testimony', 'reference']
        else:
            self.prop_types = ['Claim', 'MajorClaim', 'Premise']
        y_stub = [DocLabel(self.prop_types, [False, True])]
        self.initialize_labels(y_stub)

    def inference(self, exact=False):

        # generate stub doc
        rng = np.random.RandomState(0)
        n_props = 5
        doc = ArgDocStub(prop_types=self.prop_types,
                         n_props=n_props,
                         random_state=rng)
        # generate random potentials
        prop_potentials = rng.randn(n_props, self.n_prop_states)
        link_potentials = rng.randn(len(doc.link_to_prop), self.n_link_states)
        compat_potentials = rng.randn(self.n_prop_states,
                                      self.n_prop_states,
                                      self.n_link_states)
        grandparent_potentials = rng.randn(len(doc.second_order))
        coparent_potentials = sibling_potentials = []
        potentials = (prop_potentials,
                      link_potentials,
                      compat_potentials,
                      coparent_potentials,
                      grandparent_potentials,
                      sibling_potentials)

        return self._inference(doc, potentials, return_energy=True,
                               constraints=self.constraints,
                               exact=exact)

@pytest.mark.parametrize('constraints', [
    'none',
    'ukp',
    'ukp-strict',
    'cdcp',
    'cdcp-strict'
])
@pytest.mark.parametrize('exact', [False, True])
def test_smoke_inference(constraints, exact):
    # test that inferece runs without errors
    y_hat, status, energy = InferenceStub(constraints).inference(exact)
