"""Merlin: 3-D abdominal CT, organ masks O from TotalSegmentator, body = TotalSegmentator trunk.
s_f(x) = <phi(x), psi_f>."""
import json, os
import numpy as np
from scipy.ndimage import binary_dilation
from visualdependence import Case, VLMAdapter
from . import merlin_common as C

PLANE, AXIS, OFFSETS = (0, 1), 0, [-20, -10, 0, 10, 20]
GUARD_SAMPLE = 150


class MerlinAdapter(VLMAdapter):
    def __init__(self, model, psi, device):
        import torch
        self.torch, self.model, self.psi, self.device = torch, model, psi, device

    def embed_image(self, image):
        t = self.torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(self.device)
        with self.torch.no_grad():
            e = self.model(t, ["x"])[0][0]
            e = e.as_tensor() if hasattr(e, "as_tensor") else e
        return (e / (e.norm() + 1e-8)).float().cpu().numpy()

    def score(self, embedding, finding):
        return float(np.asarray(embedding) @ self.psi[finding])


def findings():
    return [f for f in C.FINDING_ORGAN if f not in C.EXCLUDED]


def load(random_encoder=False):
    """Merlin with its text vectors; random_encoder resets every weight of the image encoder (negative control)."""
    import torch
    # from visualdependence.adapters.merlin import Merlin
    from merlin import Merlin 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Merlin(ImageEmbedding=False).to(device).eval()
    if random_encoder:
        torch.manual_seed(0)
        for m in model.model.encode_image.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    return MerlinAdapter(model, {f: C.encode_text(model, C.PROMPTS[f], device) for f in findings()}, device)


def load_volume(sid):
    from merlin.data import DataLoader
    dl = DataLoader(datalist=[{"image": f"{C.IMAGES}/{sid}.nii.gz", "text": "x"}], cache_dir=C.CACHE,
                    batchsize=1, shuffle=False, num_workers=0)
    return next(iter(dl))["image"][0, 0].cpu().numpy()


def make_case(sid, labels):
    body = C.load_mask(sid, ["body_trunc"])
    if not body.any():
        return None
    x = load_volume(sid)
    masks, y, region_of = {}, {}, {}
    for f in findings():
        if labels.get(f) not in (0, 1):
            continue
        O = C.load_mask(sid, C.FINDING_ORGAN[f])
        if not O.any():
            continue
        r = C.margin(int(O.sum()))
        O = (binary_dilation(O, iterations=r) if r else O) & body
        if O.any():
            masks[f], y[f], region_of[f] = O, int(labels[f]), "|".join(C.FINDING_ORGAN[f])
    return Case(image=x, masks=masks, y=y, body=body, region_of=region_of) if masks else None


def cases(scans=None):
    """(study id, Case) for every scan with images, labels and masks; `scans` is an optional list of study ids."""
    import pandas as pd
    lab = pd.read_csv(C.LABELS)
    lab = lab.set_index(lab.columns[0]); lab.index = lab.index.astype(str)
    ids = scans if scans is not None else sorted(p.name for p in C.MASKS.iterdir() if (p / "body_trunc.nii.gz").exists())
    for sid in ids:
        if sid in lab.index and os.path.exists(f"{C.IMAGES}/{sid}.nii.gz"):
            case = make_case(sid, lab.loc[sid].to_dict())
            if case is not None:
                yield sid, case
