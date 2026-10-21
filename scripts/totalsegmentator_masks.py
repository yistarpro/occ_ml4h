"""The Merlin masks: TotalSegmentator v2 organ structures (task total, --fast) and the body trunk (task body).

    python scripts/totalsegmentator_masks.py [study ids ...]
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adapters.merlin_common import STRUCTURES
from config.paths import MERLIN_IMAGES as images, MERLIN_MASKS as masks, TOTALSEGMENTATOR

ids = sys.argv[1:] or sorted(f[:-7] for f in os.listdir(images) if f.endswith(".nii.gz"))
for sid in ids:
    ct, out = f"{images}/{sid}.nii.gz", f"{masks}/{sid}"
    if not os.path.exists(f"{out}/colon.nii.gz"):
        subprocess.run([TOTALSEGMENTATOR, "-i", ct, "-o", out, "--fast", "--roi_subset", *STRUCTURES], check=True)
    if not os.path.exists(f"{out}/body_trunc.nii.gz"):
        subprocess.run([TOTALSEGMENTATOR, "-i", ct, "-o", out, "--task", "body"], check=True)
