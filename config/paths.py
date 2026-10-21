"""Every path the audit reads. Edit once."""

# Merlin
MERLIN_IMAGES = "/path/to/merlin_data"                               # <study id>.nii.gz
MERLIN_LABELS = "/path/to/zero_shot_findings_disease_cls.csv"
MERLIN_MASKS = "/path/to/merlin_masks"                               # <study id>/<structure>.nii.gz, written by scripts/totalsegmentator_masks.py
MERLIN_CACHE = "/path/to/merlin_cache"                               # Merlin's loader cache, any writable folder
TOTALSEGMENTATOR = "TotalSegmentator"                                # the TotalSegmentator command

# RenalCLIP
KITS23_DIR = "/path/to/kits23/dataset"                               # case_XXXXX/{imaging,segmentation}.nii.gz, kits23.json
RENALCLIP_DIR = "/path/to/RenalCLIP"                                 # clone of github.com/dt-yuhui/RenalCLIP
RENALCLIP_CKPT = "/path/to/RenalCLIP-image-encoder-model-best-acc.pt"
LLAMA3_DIR = "/path/to/Meta-Llama-3-8B-Instruct"
RENALCLIP_LLM2VEC = "/path/to/RenalCLIP/checkpoints/LLM2Vec"         # the MNTP and SimCSE LoRA adapters
RENALCLIP_TEXT_VECS = "/path/to/renalclip_text_vecs.npz"             # written on first use

# MedSigLIP
MEDSIGLIP_SNAPSHOT = "google/medsiglip-448"                          # or a local snapshot folder
CHESTXRAY14_DIR = "/path/to/chestxray14"                             # images/, BBox_List_2017.csv, Data_Entry_2017_v2020.csv, test_list.txt
