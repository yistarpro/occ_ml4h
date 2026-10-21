"""RenalCLIP: its tumor-centered one-kidney crop, its image encoder and its prompts."""
import os, sys
from collections import OrderedDict
import numpy as np

from config.paths import (KITS23_DIR as KITS, RENALCLIP_DIR, RENALCLIP_CKPT as CKPT, LLAMA3_DIR as LLAMA3,
                          RENALCLIP_LLM2VEC as LORA, RENALCLIP_TEXT_VECS as TEXT_VECS)

# The prompts: 20 templates x 5 attributes per class.
TEMPLATES = [
    "A soft tissue mass that is ____ is visible in the kidney.", "A density shadow that is ____ is observed in the kidney.",
    "An area that is ____ is visualized at the pole of the kidney.", "A mass-like shadow that is ____ is noted in the kidney.",
    "An occupying lesion that is ____ is seen in the kidney.", "An abnormality that is ____ is detected in the kidney.",
    "A region of interest that is ____ is found in the kidney.", "A structural irregularity that is ____ is visible in the kidney.",
    "A mixed-density lesion that is ____ is apparent in the kidney.", "A nodule that is ____ is observed in the kidney.",
    "A shadow-like lesion that is ____ is identified in the kidney.", "A well-defined region that is ____ is noted in the kidney.",
    "An abnormal soft tissue that is ____ is detected in the kidney.", "A feature that is ____ is recognized in the kidney.",
    "A density lesion that is ____ is revealed in the kidney.", "The kidney shows a soft tissue mass that is ____.",
    "The kidney reveals a density shadow that is ____.", "The kidney demonstrates a lesion that is ____.",
    "The kidney presents an occupying abnormality that is ____.", "The kidney displays a structural irregularity that is ____.",
]
ATTRIBUTES = {
    "benign": [
        "benign, described as a mass containing fatty soft tissue, consistent with an angiomyolipoma",
        "benign, presenting as a round cystic lesion with a clear boundary and no enhancement after contrast",
        "benign, a finding of a nodule with uniform density, smooth borders, and no significant enhancement",
        "benign, a complex cyst with fine septa showing only mild, non-suspicious enhancement",
        "benign, a lesion showing mild and uniform progressive enhancement after contrast, without washout"],
    "malignant": [
        "malignant, showing early significant uneven enhancement during the corticomedullary phase, and reduced density in the nephrographic phase",
        "malignant, presenting as a mass-like abnormal density shadow which is uneven and contains a central necrotic area",
        "malignant, a cystic-solid occupying lesion with thickened, irregular walls and enhancing solid nodules",
        "malignant, described as a solid tumor showing rapid enhancement and subsequent washout of contrast",
        "malignant, an irregular mass with unclear boundaries and heterogeneous density"],
}


def text_vectors():
    """psi_malignant and psi_benign: the unit mean embedding of each class's 100 prompts (LLM2Vec on Llama-3).
    Computed once and stored in RENALCLIP_TEXT_VECS."""
    if os.path.exists(TEXT_VECS):
        z = np.load(TEXT_VECS)
        key = {"malignant": "malignant" if "malignant" in z else "avg_1", "benign": "benign" if "benign" in z else "avg_0"}
        return {k: z[v] / np.linalg.norm(z[v]) for k, v in key.items()}
    import json, dataclasses, torch
    from llm2vec import LLM2Vec
    from peft import LoraConfig
    mntp = f"{LORA}/Meta-Llama-3-8B-Instruct-radiology-ext-long"
    simcse = f"{LORA}/Meta-Llama-3-8B-Instruct-radiology-simcse/checkpoint-1000"
    valid = {f.name for f in dataclasses.fields(LoraConfig)}
    for d in (mntp, simcse):                                   # adapter configs readable by peft 0.10.0
        p = f"{d}/adapter_config.json"; c = json.load(open(p))
        c = {k: v for k, v in c.items() if k in valid}; c["base_model_name_or_path"] = LLAMA3
        json.dump(c, open(p, "w"), indent=2)
    l2v = LLM2Vec.from_pretrained(base_model_name_or_path=LLAMA3, peft_model_name_or_path=mntp, device_map="cuda",
                                  torch_dtype=torch.float16, pooling_mode="mean", max_length=224)
    l2v.model.load_adapter(simcse, adapter_name="simcse"); l2v.model.set_adapter("simcse")
    out = {}
    for cls, attrs in ATTRIBUTES.items():
        e = [np.asarray(l2v.encode([t.replace("____", a)])).squeeze() for a in attrs for t in TEMPLATES]
        e = np.stack([v / np.linalg.norm(v) for v in e]).mean(0)
        out[cls] = (e / np.linalg.norm(e)).astype(np.float32)
    np.savez(TEXT_VECS, **out)
    return out


def build_encoder(device="cuda"):
    import torch
    sys.path.insert(0, RENALCLIP_DIR)
    from models.resnet import resnet18
    enc = resnet18(num_classes=2, shortcut_type="A", mode="pretrain", clip_hidden_dim=None, clip_output_dim=4096)
    ck = torch.load(CKPT, map_location="cpu")["model"]
    pre = "module.image_encoder_q_student."
    enc.load_state_dict(OrderedDict((k[len(pre):], v) for k, v in ck.items() if k.startswith(pre)), strict=False)
    return enc.to(device).eval()


def _box(a, c, half, cval):
    X, Y, Z = a.shape
    lo = [int(round(c[i] - half[i])) for i in range(3)]; hi = [int(round(c[i] + half[i])) for i in range(3)]
    out = np.full(tuple(h - l for l, h in zip(lo, hi)), cval, a.dtype)
    s = [slice(max(l, 0), min(h, n)) for l, h, n in zip(lo, hi, (X, Y, Z))]
    d = [slice(si.start - l, si.stop - l) for si, l in zip(s, lo)]
    out[tuple(d)] = a[tuple(s)]
    return out


def _resize(a, size, mode):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(a.astype(np.float32))[None, None]
    t = F.interpolate(t, size=size, mode="nearest") if mode == "nearest" else F.interpolate(t, size=size, mode=mode, align_corners=False)
    return t[0, 0].numpy()


def preprocess(case_id):
    """Crop 140x140x160 mm around the tumor centroid, resize to 140x140x32, window WL 50 / WW 500,
    center-crop to 128x128x32. Returns the volume (32,128,128) in [0,1] and the segmentation in the same frame."""
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
    d = Compose([LoadImaged(keys=["image", "seg"]), EnsureChannelFirstd(keys=["image", "seg"]),
                 Orientationd(keys=["image", "seg"], axcodes="RAS")])({"image": f"{KITS}/{case_id}/imaging.nii.gz",
                                                                         "seg": f"{KITS}/{case_id}/segmentation.nii.gz"})
    img = d["image"][0].numpy().astype(np.float32); seg = d["seg"][0].numpy().astype(np.int16)
    sx, sy, sz = [float(p) for p in d["image"].pixdim[:3]]
    t = np.argwhere(seg == 2)
    t = t if len(t) else np.argwhere(seg == 3) if (seg == 3).any() else np.argwhere(seg == 1)
    half = (70.0 / sx, 70.0 / sy, 80.0 / sz)
    img = _resize(_box(img, t.mean(0), half, -1024.0), (140, 140, 32), "trilinear")
    seg = _resize(_box(seg, t.mean(0), half, 0), (140, 140, 32), "nearest")
    img = np.clip((img + 200.0) / 500.0, 0.0, 1.0)
    cc = lambda a: a[6:134, 6:134, :]
    return np.transpose(cc(img), (2, 0, 1)).astype(np.float32), np.transpose(cc(seg), (2, 0, 1)).round().astype(np.int16)


def embed(enc, vol, device="cuda"):
    import torch, torch.nn.functional as F
    x = torch.from_numpy(vol)[None, None].to(device)
    with torch.no_grad():
        f = enc.global_embedding(enc.avgpool(enc.forward_features(x)).flatten(1))
    return F.normalize(f, dim=-1)[0].cpu().numpy().astype(np.float32)
