# Dependence Audit Without Maps: An Occlusion Method for Medical Vision–Language Models

This is the implementation of the occlusion audit in "Dependence Audit Without Maps: An Occlusion Method for Medical Vision–Language Models" (ML4H 2026, Findings track). The audit tests whether a vision–language model's score for a finding depends on the finding's anatomy, without producing a saliency map: it erases the finding's region, measures the change in the model's score, and compares it with the same erasure applied to matched control regions.

The audit applies to dual-encoder VLMs, which score an image $x$ against a finding $f$ by the inner product of a unit image embedding and a unit text embedding, $s_f(x)$. A case carries a binary label $y_f$ for each finding and a segmentation mask $O$ of that finding's organ or lesion.

## The pipeline

1. **Score.** $s_f(x)=\langle\phi(x),\psi_f\rangle$ (`VLMAdapter.score`).
2. **Erase the region and $K{=}2$ matched controls.** Erasing $O$ by filling it with the body's mean intensity lowers the score by $d_{\mathrm{organ}}(x,f)=s_f(x)-s_f(x_O)$. Each control is a rigid copy of $O$ placed elsewhere in the body at the same relative eccentricity (`controls.rigid_controls`); $d_{\mathrm{control}}$ is their mean drop (`audit.drops`).
3. **Gates.** A case–finding pair is reported only if it passes three gates: an on-manifold guard, which rejects a perturbation that takes the image off the data distribution (`validity.OnManifoldGuard`); a containment gate, which excludes findings whose evidence lies outside the mask being erased (`EXCLUDED` in `adapters/merlin_common.py`); and a constructibility gate, which reports a case as N/A when $K$ copies of the region do not fit.
4. **Measures** (`visualdependence/audit.py`, arithmetic in `visualdependence/measures.py`):

| question | measure | function |
|---|---|---|
| Does the score depend on the region? (dependence) | $\mathrm{occ}=(d_{\mathrm{organ}}-d_{\mathrm{control}})/\mathrm{std}_{\mathrm{null}}$, larger | `dependence` |
| Does the dependence follow the anatomy? (position) | $\mathrm{pos}$, smaller | `position_terms`, `position` |
| Is it specific to the finding? (finding-specificity) | $R_O$, larger | `finding_specificity` |
| Is it specific to the label? (label-specificity) | $G_f$, larger | `label_specificity` |

$\mathrm{std}_{\mathrm{null}}$ is the standard deviation of a null in which one control takes the region's place (`measures.null`, `measures.std_null`).

## Layout

- `visualdependence/`: the audit (`measures`, `controls`, `perturb`, `validity`, `adapter`, `audit`).
- `adapters/`: Merlin (abdominal CT, TotalSegmentator organ masks), RenalCLIP (KiTS23 tumor masks), MedSigLIP (ChestX-ray14 radiologist boxes), with each model's prompts; `merlin_grounded.py` is the positive control, `run.py merlin --random-encoder` the negative control.
- `config/paths.py`: every path.
- `run_audit.sh`: the whole audit of one model.
- `run.py`: one step for one model (`drops`, `guard_fit`, `guard`, `position`).
- `summarize.py`: the four measures from a model's tables.
- `scripts/totalsegmentator_masks.py`: the Merlin organ masks and body trunk.
- `examples/toy.py`: the audit on a toy model, no data needed; `examples/adapter_template.py`: wrapping a new model.

## Running

    pip install -r requirements.txt && pip install -e .
    python examples/toy.py

Set the paths in `config/paths.py`, then run the whole audit of one model:

    bash run_audit.sh merlin          # or renalclip, medsiglip, validation

The results are stored in `results/<model>/`: the intermediate tables (`drops.csv`, `guard.npz`, `guard.csv`, `position.json`) and the four measures in `summary.json`, which also holds, per finding, $\overline{\mathrm{occ}}$ over the present pairs, the separation left after erasing the region, the detection AUROC after erasing the region and after erasing the controls, and the correlations with $\Delta_f$ (the appendix analyses of the paper). For `validation`, each scorer has its own folder, `results/validation/randomized_encoder/` and `results/validation/grounded_scorer/`. A second argument sets another output folder.

For Merlin, `run_audit.sh` uses the scans of `data/merlin_scans.json` if the file is present, and every scan in
`MERLIN_IMAGES` otherwise, with a warning. It runs, step by step:

    python scripts/totalsegmentator_masks.py data/merlin_scans.json
    python run.py merlin drops     --scans data/merlin_scans.json --out drops.csv
    python run.py merlin guard_fit --scans data/merlin_scans.json --out guard.npz
    python run.py merlin guard     --scans data/merlin_scans.json --guard-fit guard.npz --drops drops.csv --out guard.csv
    python run.py merlin position  --scans data/merlin_scans.json --guard-fit guard.npz --out position.json
    python summarize.py --drops drops.csv --guard guard.csv --position position.json

Without `--scans`, a step runs on every case; the mask script also takes study ids or no argument.
RenalCLIP and MedSigLIP run the same four steps without masks or a scan list, and their position step fits its guard on
the cases it shifts, with the first case's blanked image as the reference. MedSigLIP's boxes are all present cases, so
its detection AUROC comes from `python run.py medsiglip detection --out detection.csv`, passed to
`summarize.py --detection detection.csv`. For the validation of the measure design, the randomized encoder runs
`drops`, `guard_fit` and `guard` with `--random-encoder`, and `merlin_grounded` runs `drops` only.
On a cluster, `--shard i/N` splits the cases of any step; give `summarize.py` every shard's files.

Every path is set once in `config/paths.py`. Sources:

| model | sources |
|---|---|
| Merlin | `pip install merlin-vlm`; the Merlin Abdominal CT Dataset (Stanford AIMI, data use agreement); masks from TotalSegmentator v2 |
| RenalCLIP | github.com/dt-yuhui/RenalCLIP and huggingface.co/taoyh/RenalCLIP; Meta-Llama-3-8B-Instruct; KiTS23 (github.com/neheller/kits23) |
| MedSigLIP | huggingface.co/google/medsiglip-448; ChestX-ray14 images, `BBox_List_2017.csv`, `Data_Entry_2017_v2020.csv`, `test_list.txt` |

## Software

Python 3.11, CUDA 12.1. `pip install -r requirements.txt` for the audit; TotalSegmentator runs in a separate environment,
`pip install -r requirements-totalsegmentator.txt`.
