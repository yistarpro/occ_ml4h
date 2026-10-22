"""RenalCLIP: 3-D kidney CT (KiTS23), the tumor mask as O, one finding (malignancy) with a diagnostic label.
s(x) = <phi(x), psi_malignant> - <phi(x), psi_benign>."""
import json, os
import numpy as np
from visualdependence import Case, VLMAdapter
from . import renalclip_common as C

PLANE, AXIS, OFFSETS = (1, 2), 1, [-24, -12, 0, 12, 24]
GUARD_SAMPLE = 150
POSITION_GUARD = "cohort"
FINDING = "malignancy"


class RenalCLIPAdapter(VLMAdapter):
    def __init__(self, enc, psi, device):
        self.enc, self.psi, self.device = enc, psi, device

    def embed_image(self, image):
        return C.embed(self.enc, np.ascontiguousarray(image, dtype=np.float32), self.device)

    def score(self, embedding, finding):
        e = np.asarray(embedding)
        return float(e @ self.psi["malignant"] - e @ self.psi["benign"])


def load(random_encoder=False):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RenalCLIPAdapter(C.build_encoder(device), C.text_vectors(), device)


def cases(scans=None):
    """(case id, Case) for every KiTS23 case with a malignancy label; the body is the crop above 0.03."""
    meta = {m["case_id"]: m["malignant"] for m in json.load(open(f"{C.KITS}/kits23.json"))}
    for c in (scans or sorted(meta)):
        if meta.get(c) not in (True, False) or not os.path.exists(f"{C.KITS}/{c}/segmentation.nii.gz"):
            continue
        x, seg = C.preprocess(c)
        O = seg == 2
        if O.any():
            y = int(meta[c])
            yield c, Case(image=x, masks={FINDING: O}, y={FINDING: y}, body=x > 0.03, region_of={FINDING: "kidney"}, class_label=y)
