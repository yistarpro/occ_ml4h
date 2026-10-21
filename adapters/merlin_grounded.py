"""Positive control: a scorer that reads the mask directly,
s_f(x) = 1/|O_f| sum_{v in O_f} |x_v - mu_B| + lambda 1/|B| sum_{v in B} |x_v - mu_B|, lambda = 0.05.
Same scans, masks and control rule as Merlin; it uses no embedding, so the on-manifold guard does not apply."""
import numpy as np
from visualdependence import VLMAdapter
from .merlin import PLANE, AXIS, OFFSETS, GUARD_SAMPLE, cases as _cases

LAMBDA = 0.05


class GroundedScorer(VLMAdapter):
    case = None

    def embed_image(self, image):
        return np.asarray(image, np.float32)

    def score(self, x, finding):
        body = x[self.case.body]; mu = float(body.mean())
        return float(np.abs(x[self.case.masks[finding]] - mu).mean() + LAMBDA * np.abs(body - mu).mean())


_scorer = GroundedScorer()


def load(random_encoder=False):
    return _scorer


def cases(scans=None):
    for sid, case in _cases(scans):
        _scorer.case = case
        yield sid, case
