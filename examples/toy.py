"""The audit on a toy model whose score reads one region, so the answer is known: occ > 0, R_O > 1."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from visualdependence import VLMAdapter, Case, drops, std_null, dependence, finding_specificity

N = 96
yy, xx = np.mgrid[:N, :N]
BODY = (yy - N / 2) ** 2 + (xx - N / 2) ** 2 < 40 ** 2
LUNG = (yy - 30) ** 2 / 8 ** 2 + (xx - 66) ** 2 / 12 ** 2 < 1                 # one organ mask O
SPOTS = {"nodule": (28, 62), "mass": (32, 70)}                                # two findings on it


class Toy(VLMAdapter):
    """phi(x): the mean intensity around each finding's spot; s_f reads its own spot."""
    def embed_image(self, x):
        return np.array([x[r - 2:r + 3, c - 2:c + 3].mean() for r, c in SPOTS.values()] + [x[BODY].mean()])

    def score(self, e, f):
        i = list(SPOTS).index(f)
        return float(e[i] - (1.0 + 0.5 * i) * e[-1])            # the findings weigh the rest of the body differently


rng = np.random.default_rng(0)
pairs = []
for n in range(60):
    y = {"nodule": int(n % 3 == 0), "mass": int(n % 3 == 1)}
    x = (0.5 + rng.normal(0, 0.05, (N, N))) * BODY
    for f, (r, c) in SPOTS.items():
        x[r - 2:r + 3, c - 2:c + 3] += 0.4 * y[f]
    case = Case(image=x, masks={"nodule": LUNG, "mass": LUNG}, y=y, body=BODY, region_of={"nodule": "lung", "mass": "lung"})
    pairs += drops(Toy(), case, K=2, case_id=n)[0]

sn = std_null(pairs)
print("dependence", dependence(pairs, sn))
print("finding-specificity", finding_specificity(pairs, sn))
