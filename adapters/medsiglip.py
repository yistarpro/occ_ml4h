"""MedSigLIP: 2-D chest radiographs (ChestX-ray14), the radiologist box as O, at the native 1024 px.
s_f(x) = <phi(x), psi_f^+> - <phi(x), psi_f^->. The body is the whole image."""
import os
import numpy as np
from visualdependence import Case, VLMAdapter
from . import medsiglip_common as C

PLANE, AXIS, OFFSETS = (0, 1), 1, [-96, -48, 0, 48, 96]
GUARD_SAMPLE = 200
POSITION_GUARD = "cohort"       


class MedSigLIPAdapter(VLMAdapter):
    def __init__(self, model, proc, psi, device):
        self.model, self.proc, self.psi, self.device = model, proc, psi, device

    def embed_image(self, image):
        return C.embed_image(self.model, self.proc, image, self.device)

    def score(self, embedding, finding):
        pos, neg = self.psi[finding]
        e = np.asarray(embedding)
        return float(e @ pos - e @ neg)


def load(random_encoder=False):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = C.load(device)
    return MedSigLIPAdapter(model, proc, C.text_vectors(model, proc, device), device)


def _box_mask(bx, shape):
    m = np.zeros(shape, bool)
    for x, y, w, h in bx:
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(shape[1], int(round(x + w))), min(shape[0], int(round(y + h)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def cases(scans=None):
    """("image|finding", Case) for every box; `scans` may list image names or "image|finding" case ids (the guard
    step passes the drop table's case ids). A control must also avoid the other findings' boxes on the image.
    The box exists only where the finding is annotated, so every case is present."""
    from PIL import Image
    groups = C.boxes()
    for im, f in sorted(groups):
        if f not in C.PHRASE or (scans and im not in scans and f"{im}|{f}" not in scans) or not os.path.exists(f"{C.CXR14}/images/{im}"):
            continue
        x = np.asarray(Image.open(f"{C.CXR14}/images/{im}").convert("L"), np.float32)
        O = _box_mask(groups[(im, f)], x.shape)
        other = np.zeros(x.shape, bool)
        for (im2, f2), bx in groups.items():
            if im2 == im and f2 != f:
                other |= _box_mask(bx, x.shape)
        if O.any():
            yield f"{im}|{f}", Case(image=x, masks={f: O}, y={f: 1}, body=np.ones(x.shape, bool),
                                    exclude=other if other.any() else None)


def detection(adapter, cap=15000):
    """Zero-shot detection AUROC of each finding on the image-level labels of the test split (a 15,000-image sample)."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from PIL import Image
    meta = pd.read_csv(f"{C.CXR14}/Data_Entry_2017_v2020.csv")
    meta = meta.rename(columns={meta.columns[0]: "image", "Finding Labels": "labels"})
    test = set(open(f"{C.CXR14}/test_list.txt").read().split())
    sub = meta[meta.image.isin(test) & meta.image.isin(set(os.listdir(f"{C.CXR14}/images")))].reset_index(drop=True)
    if cap and len(sub) > cap:
        sub = sub.sample(cap, random_state=0).reset_index(drop=True)
    E = np.stack([adapter.embed_image(np.asarray(Image.open(f"{C.CXR14}/images/{im}").convert("L"), np.float32)) for im in sub.image])
    rows = []
    for f in C.PHRASE:
        y = sub["labels"].str.split("|").apply(lambda v: f in v).astype(int).values
        s = E @ adapter.psi[f][0] - E @ adapter.psi[f][1]
        rows.append(dict(finding=f, n_pos=int(y.sum()), n=len(y), auroc=float(roc_auc_score(y, s))))
    return pd.DataFrame(rows)
