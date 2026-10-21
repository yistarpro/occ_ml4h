"""MedSigLIP (google/medsiglip-448) and the ChestX-ray14 radiologist boxes."""
import os
import numpy as np

from config.paths import MEDSIGLIP_SNAPSHOT as SNAPSHOT, CHESTXRAY14_DIR as CXR14

PHRASE = {"Atelectasis": "atelectasis", "Cardiomegaly": "an enlarged heart", "Effusion": "a pleural effusion",
          "Infiltration": "pulmonary infiltration", "Mass": "a lung mass", "Nodule": "a lung nodule",
          "Pneumonia": "pneumonia", "Pneumothorax": "a pneumothorax"}


def prompts(f):
    """The positive and negative prompt of a finding."""
    return f"a chest x-ray with {PHRASE[f]}", f"a chest x-ray with no {PHRASE[f]}"


def load(device="cuda"):
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SNAPSHOT)
    return AutoModel.from_pretrained(SNAPSHOT, torch_dtype=torch.float32).to(device).eval(), proc


def text_vectors(model, proc, device):
    """psi_f^+ and psi_f^- of every finding."""
    import torch, torch.nn.functional as F
    out = {}
    with torch.no_grad():
        for f in PHRASE:
            b = proc(text=list(prompts(f)), padding="max_length", max_length=64, return_tensors="pt").to(device)
            v = F.normalize(model.get_text_features(**b), dim=-1).cpu().numpy()
            out[f] = (v[0], v[1])
    return out


def embed_image(model, proc, image, device):
    import torch, torch.nn.functional as F
    from PIL import Image
    pil = Image.fromarray(np.clip(np.asarray(image, np.float32), 0, 255).astype(np.uint8)).convert("L").convert("RGB")
    with torch.no_grad():
        pv = proc(images=[pil], return_tensors="pt")["pixel_values"].to(device)
        return F.normalize(model.get_image_features(pixel_values=pv), dim=-1)[0].cpu().numpy()


def boxes():
    """(image, finding) -> list of (x, y, w, h); the label Infiltrate is read as Infiltration."""
    import pandas as pd
    df = pd.read_csv(f"{CXR14}/BBox_List_2017.csv")
    df.columns = ["image", "label", "x", "y", "w", "h"] + list(df.columns[6:])
    df["label"] = df["label"].str.strip().replace({"Infiltrate": "Infiltration"})
    out = {}
    for r in df.itertuples():
        out.setdefault((r.image, r.label), []).append((r.x, r.y, r.w, r.h))
    return out
