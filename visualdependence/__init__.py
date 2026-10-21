"""An audit of whether a dual-encoder VLM's finding score depends on the finding's region, without a map."""
from . import measures, controls, perturb, validity, adapter, audit
from .adapter import VLMAdapter
from .validity import OnManifoldGuard
from .audit import (Case, Pair, drops, std_null, dependence, position_terms, position, finding_specificity,
                    label_specificity)
