"""Run one step of the audit for one model.

    python run.py MODEL STEP --out FILE [--k 2] [--shard i/N] [--guard-fit FILE] [--scans FILE] [--random-encoder]

MODEL: merlin | merlin_grounded | renalclip | medsiglip
STEP:  drops      the drop table: s_f(x), d_organ and the K control drops of every case-finding pair (csv)
       guard_fit  the on-manifold guard, fitted on unperturbed images, and the blank's distance (npz)
       guard      the guard distance of the organ erasure and of each control erasure of every pair (csv)
       position   the pos terms of every present pair (json)
       detection  MedSigLIP only: detection AUROC on the image-level labels (csv)
"""
import argparse, csv, importlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualdependence import OnManifoldGuard, perturb, controls
from visualdependence.audit import drops, case_controls, position_terms

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("step", choices=["drops", "guard_fit", "guard", "position", "detection"])
ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=2)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--guard-fit"); ap.add_argument("--drops")
ap.add_argument("--scans", help="json list of case ids"); ap.add_argument("--random-encoder", action="store_true")
a = ap.parse_args()

model = importlib.import_module(f"adapters.{a.model}")
adapter = model.load(random_encoder=a.random_encoder)
scans = json.load(open(a.scans)) if a.scans else None
i, n = (int(v) for v in a.shard.split("/"))


def all_cases(ids=None):
    for k, (cid, case) in enumerate(model.cases(ids if ids is not None else scans)):
        if k % n == i:
            yield cid, case


def done_ids(path):
    """Cases already written, so that a run can resume."""
    return set(open(path + ".done").read().split()) if os.path.exists(path + ".done") else set()


if a.step == "drops":
    K = a.k; done = done_ids(a.out)
    fields = ["case", "finding", "region", "y", "class_label", "s", "d_organ"] + [f"d_control_{k + 1}" for k in range(K)]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fo, open(a.out + ".done", "a") as fd, open(a.out + ".na", "a") as fn:
        w = csv.DictWriter(fo, fieldnames=fields)
        if new:
            w.writeheader()
        for cid, case in all_cases():
            if cid in done:
                continue
            pairs, na = drops(adapter, case, K, model.PLANE, case_id=cid)
            for p in pairs:
                w.writerow(dict(case=cid, finding=p.finding, region=p.region, y=p.y, class_label="" if p.class_label is None else p.class_label,
                                s=round(p.s, 6), d_organ=round(p.d_organ, 6), **{f"d_control_{k + 1}": round(float(v), 6) for k, v in enumerate(p.d_control)}))
            for f in na:                                        # constructibility: N/A pairs
                fn.write(f"{cid},{f}\n")
            fd.write(cid + "\n"); fo.flush(); fd.flush(); fn.flush()

elif a.step == "guard_fit":
    sample = []
    for cid, case in all_cases():
        sample.append(case)
        if len(sample) == model.GUARD_SAMPLE:
            break
    guard = OnManifoldGuard().fit(np.stack([adapter.embed_image(c.image) for c in sample]))
    # The reference is a blanked image, the whole body erased.
    blank = float(np.mean([guard.distance(adapter.embed_image(perturb.erase(c.image, c.body, perturb.body_mean(c.image, c.body))))
                           for c in sample]))
    guard.save(a.out, blank)

elif a.step == "guard":
    import pandas as pd
    guard, blank = OnManifoldGuard.load(a.guard_fit)
    D = pd.read_csv(a.drops); want = D.groupby(D["case"].astype(str))["finding"].apply(set).to_dict()
    K = a.k; new = not os.path.exists(a.out); done = done_ids(a.out)
    fields = ["case", "finding", "distance_organ"] + [f"distance_control_{k + 1}" for k in range(K)] + ["blank_distance", "pass_organ", "pass_control"]
    with open(a.out, "a", newline="") as fo, open(a.out + ".done", "a") as fd:
        w = csv.DictWriter(fo, fieldnames=fields)
        if new:
            w.writeheader()
        for cid, case in all_cases(list(want)):
            if cid in done or cid not in want:
                continue
            fill = perturb.body_mean(case.image, case.body); pbody = controls.placement_body(case.body, model.PLANE)
            for f in sorted(want[cid] & set(case.masks)):
                c = case_controls(case, f, K, model.PLANE, pbody)
                if c is None:
                    continue
                do = guard.distance(adapter.embed_image(perturb.erase(case.image, case.masks[f], fill)))
                dc = [guard.distance(adapter.embed_image(perturb.erase(case.image, ck, fill))) for ck in c]
                w.writerow(dict(case=cid, finding=f, distance_organ=round(do, 4), blank_distance=round(blank, 4),
                                pass_organ=int(do < 0.5 * blank), pass_control=int(all(d < 0.5 * blank for d in dc)),
                                **{f"distance_control_{k + 1}": round(d, 4) for k, d in enumerate(dc)}))
            fd.write(cid + "\n"); fo.flush(); fd.flush()

elif a.step == "position":
    guard, blank = OnManifoldGuard.load(a.guard_fit) if a.guard_fit else (None, None)
    terms, skipped, na = [], 0, 0
    for cid, case in all_cases():
        t, s, m = position_terms(adapter, case, model.OFFSETS, model.AXIS, a.k, model.PLANE, guard, blank)
        terms += t; skipped += s; na += m
    json.dump(dict(terms=[[float(x), float(y)] for x, y in terms], skipped=skipped, na=na), open(a.out, "w"))

elif a.step == "detection":
    model.detection(adapter).to_csv(a.out, index=False)
