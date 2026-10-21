"""Merlin data: the organ mask O of each finding (TotalSegmentator), the prompts and the text vectors."""
import os
from pathlib import Path
import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, SpatialPadd, CenterSpatialCropd

from config.paths import MERLIN_IMAGES as IMAGES, MERLIN_LABELS as LABELS, MERLIN_CACHE as CACHE, MERLIN_MASKS
MASKS = Path(MERLIN_MASKS)
SHAPE = (224, 224, 160)

LUNG = ["lung_upper_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_left", "lung_lower_lobe_right"]
BOWEL = ["colon", "small_bowel", "duodenum"]
SPINE = ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", "sacrum"]
RIBS = ["rib_left_11", "rib_left_12", "rib_right_11", "rib_right_12"]

# Each finding is assigned the TotalSegmentator organ mask that holds its evidence.
FINDING_ORGAN = {
    "splenomegaly": ["spleen"], "hepatomegaly": ["liver"], "hepatic_steatosis": ["liver"], "biliary_ductal_dilation": ["liver"],
    "gallstones": ["gallbladder"], "pancreatic_atrophy": ["pancreas"],
    "renal_cyst": ["kidney_left", "kidney_right"], "renal_hypodensities": ["kidney_left", "kidney_right"],
    "hydronephrosis": ["kidney_left", "kidney_right"], "prostatomegaly": ["prostate"],
    "abdominal_aortic_aneurysm": ["aorta"], "atherosclerosis": ["aorta"], "thrombosis": ["aorta", "inferior_vena_cava"],
    "cardiomegaly": ["heart"], "coronary_calcification": ["heart"], "aortic_valve_calcification": ["heart", "aorta"],
    "pleural_effusion": LUNG, "atelectasis": LUNG, "submucosal_edema": BOWEL, "bowel_obstruction": BOWEL,
    "appendicitis": ["colon"], "hiatal_hernia": ["stomach", "esophagus"], "fracture": SPINE + RIBS, "osteopenia": SPINE,
}
# Containment gate: pleural effusion is excluded, because the fluid lies in the pleural space outside the lung mask.
EXCLUDED = {"pleural_effusion"}
STRUCTURES = sorted({s for v in FINDING_ORGAN.values() for s in v})

PROMPTS = {
    "submucosal_edema": "submucosal edema; bowel wall edema present",
    "renal_hypodensities": "renal hypodensity; hypodense renal lesion",
    "aortic_valve_calcification": "aortic valve calcification present; calcified aortic valve",
    "coronary_calcification": "coronary artery calcifications; coronary calcification present",
    "thrombosis": "thrombosis; venous thrombosis; occlusive thrombus; nonocclusive thrombus",
    "metastatic_disease": "metastatic disease; metastases present",
    "pancreatic_atrophy": "pancreatic atrophy; atrophic pancreas",
    "renal_cyst": "renal cyst; simple renal cyst",
    "osteopenia": "osteopenia; low bone density",
    "surgically_absent_gallbladder": "surgically absent gallbladder; post cholecystectomy",
    "atelectasis": "atelectasis; subsegmental atelectasis",
    "abdominal_aortic_aneurysm": "abdominal aortic aneurysm; infrarenal AAA",
    "anasarca": "anasarca; diffuse body wall edema",
    "hiatal_hernia": "hiatal hernia",
    "lymphadenopathy": "lymphadenopathy; enlarged lymph nodes",
    "prostatomegaly": "prostatomegaly; enlarged prostate",
    "biliary_ductal_dilation": "biliary ductal dilatation; dilated bile ducts; intrahepatic biliary dilatation",
    "cardiomegaly": "cardiomegaly; enlarged heart",
    "splenomegaly": "splenomegaly; enlarged spleen",
    "hepatomegaly": "hepatomegaly; enlarged liver",
    "atherosclerosis": "atherosclerosis; atherosclerotic calcifications",
    "ascites": "ascites; free intraperitoneal fluid",
    "pleural_effusion": "pleural effusion; left pleural effusion; right pleural effusion",
    "hepatic_steatosis": "hepatic steatosis; fatty liver",
    "appendicitis": "appendicitis; inflamed appendix",
    "gallstones": "gallstones; cholelithiasis",
    "hydronephrosis": "hydronephrosis; pelvicaliectasis",
    "bowel_obstruction": "bowel obstruction; small bowel obstruction",
    "free_air": "free air; pneumoperitoneum",
    "fracture": "fracture; vertebral fracture; rib fracture",
}

_mask_tf = Compose([LoadImaged(keys=["m"]), EnsureChannelFirstd(keys=["m"]), Orientationd(keys=["m"], axcodes="RAS"),
                    Spacingd(keys=["m"], pixdim=(1.5, 1.5, 3), mode="nearest"), SpatialPadd(keys=["m"], spatial_size=list(SHAPE)),
                    CenterSpatialCropd(keys=["m"], roi_size=list(SHAPE))])


def load_mask(sid, structures):
    """Union of TotalSegmentator structures, in the frame of Merlin's input volume."""
    m = np.zeros(SHAPE, bool)
    for s in structures:
        p = MASKS / sid / f"{s}.nii.gz"
        if p.exists():
            m |= _mask_tf({"m": str(p)})["m"][0].numpy() > 0.5
    return m


def margin(n_voxels):
    """The size-dependent dilation of the organ mask, in voxels."""
    return 0 if n_voxels < 8192 else 1 if n_voxels < 32768 else 2


def encode_text(model, phrase, device):
    """psi_f: the unit text embedding of the finding's prompt."""
    import torch
    with torch.no_grad():
        e = model.model.encode_text([phrase]).float()
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        v = e.mean(0)
    return (v / (v.norm() + 1e-8)).cpu().numpy()
