"""The four measures of one model from its tables.

    python summarize.py --drops D.csv [D2.csv ...] [--guard G.csv ...] [--position P.json ...] [--detection A.csv] [--out S.json]

Detection AUROC is taken from the pairs, or from --detection (finding, auroc) where the pairs have one class only.

A pair is kept only if its organ erasure and every control erasure pass the on-manifold guard."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualdependence import Pair, std_null, dependence, position, finding_specificity, label_specificity
from visualdependence.measures import auroc, occ, spearman_ci

ap = argparse.ArgumentParser()
ap.add_argument("--drops", nargs="+", required=True); ap.add_argument("--guard", nargs="*", default=[])
ap.add_argument("--position", nargs="*", default=[]); ap.add_argument("--detection"); ap.add_argument("--out")
a = ap.parse_args()

D = pd.concat([pd.read_csv(p) for p in a.drops]).drop_duplicates(["case", "finding"])
D["case"] = D["case"].astype(str)
if a.guard:
    G = pd.concat([pd.read_csv(p) for p in a.guard]); G["case"] = G["case"].astype(str)
    fail = set(map(tuple, G[(G.pass_organ == 0) | (G.pass_control == 0)][["case", "finding"]].values))
    D = D[~D[["case", "finding"]].apply(tuple, axis=1).isin(fail)]
dc = [c for c in D.columns if c.startswith("d_control_")]
cl = D["class_label"] if "class_label" in D else pd.Series([np.nan] * len(D), index=D.index)
pairs = [Pair(r.case, r.finding, r.region, int(r.y), float(r.s), float(r.d_organ), np.array([getattr(r, c) for c in dc], float),
              None if pd.isna(l) else int(l)) for r, l in zip(D.itertuples(), cl)]

sn = std_null(pairs)
out = dict(pairs=len(pairs), present=sum(p.y == 1 for p in pairs), std_null=sn,
           dependence=dependence(pairs, sn),
           finding_specificity=finding_specificity(pairs, sn),
           label_specificity=label_specificity(pairs, sn),
           detection_auroc={f: auroc(g.s, g.y) for f, g in D.groupby("finding")})
auc = {f: v for f, v in out["detection_auroc"].items() if np.isfinite(v)}
if a.detection:
    auc = pd.read_csv(a.detection).set_index("finding")["auroc"].to_dict()
occ_f = {}
for f in sorted({p.finding for p in pairs}):
    fp = [p for p in pairs if p.finding == f and p.y == 1]
    if fp:
        occ_f[f] = float(occ(np.array([p.d_organ for p in fp]), np.stack([p.d_control for p in fp]), sn).mean())
ls = out["label_specificity"]
both = {f: r for f, r in ls.items() if r["n1"] > 0 and r["n0"] > 0}
quantities = {"occ_bar": occ_f, "gap": {f: r["gap"][0] for f, r in both.items()}, "occ_bar_y0": {f: r["occ_y0"][0] for f, r in both.items()},
              "G_f": {f: r["G_f"][0] for f, r in both.items() if r["established"]}, "Delta_f": {f: r["Delta_f"][0] for f, r in both.items()}}
out["occ_bar"] = occ_f

# Detection after erasure: AUROC of s - d_organ and of s - mean d_control, per finding with both classes.
D["s_organ_erased"] = D.s - D.d_organ
D["s_control_erased"] = D.s - D[dc].mean(axis=1)
out["detection_after_erasure"] = {}
for f, g in D.groupby("finding"):
    if 0 < g.y.sum() < len(g):
        out["detection_after_erasure"][f] = dict(n1=int(g.y.sum()), n0=int(len(g) - g.y.sum()), none=auroc(g.s, g.y),
                                                 organ=auroc(g.s_organ_erased, g.y), control=auroc(g.s_control_erased, g.y))
big = {f: r for f, r in out["detection_after_erasure"].items() if r["n1"] >= 5}
if big:
    out["detection_after_erasure_mean"] = dict(n=len(big), **{k: float(np.mean([r[k] for r in big.values()])) for k in ("none", "organ", "control")},
                                               organ_below_control=sum(r["organ"] - r["none"] < r["control"] - r["none"] for r in big.values()))

# Correlations with Delta_f, over the findings whose separation is established.
est = {f: r for f, r in both.items() if r["established"]}
out["correlation_with_delta_f"] = {}
for q, key in (("G_f", "G_f"), ("gap", "gap")):
    if len(est) >= 4:
        out["correlation_with_delta_f"][q] = spearman_ci([r[key][0] for r in est.values()], [r["Delta_f"][0] for r in est.values()])

out["correlation_with_auroc"] = {}
for q, v in quantities.items():
    ks = [f for f in v if f in auc and np.isfinite(auc[f])]
    if len(ks) >= 4:                                            # Spearman over findings
        out["correlation_with_auroc"][q] = spearman_ci([v[f] for f in ks], [auc[f] for f in ks])
if a.position:
    P = [json.load(open(p)) for p in a.position]
    out["position"] = dict(position([tuple(t) for p in P for t in p["terms"]]), skipped=sum(p["skipped"] for p in P), na=sum(p["na"] for p in P))

print(f"pairs {out['pairs']} (present {out['present']}), std_null {sn:.4f}")
d = out["dependence"]; print(f"dependence      occ_bar = {d['occ_bar']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}], n = {d['n']}")
if "position" in out:
    q = out["position"]; print(f"position        pos = {q['pos']:.2f} [{q['ci'][0]:.2f}, {q['ci'][1]:.2f}], n = {q['n']}")
for O, r in out["finding_specificity"].items():
    print(f"finding-spec.   R_O({O}) = " + (f"{r['R_O']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}], n = {r['n']}" if r["R_O"] is not None else f"not reported, n = {r['n']} < 5"))
for f, r in out["label_specificity"].items():
    print(f"label-spec.     G_f({f}) = {r['G_f'][0]:+.2f} [{r['G_f'][1]:+.2f}, {r['G_f'][2]:+.2f}]" + ("" if r["established"] else "  (not established)"))
for q, r in out["correlation_with_auroc"].items():
    print(f"rho({q}, AUROC) = {r['rho']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}], p = {r['p']:.2f}, n = {r['n']}")
for q, r in out["correlation_with_delta_f"].items():
    print(f"rho({q}, Delta_f) = {r['rho']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}], p = {r['p']:.2f}, n = {r['n']}")
for f, r in out["detection_after_erasure"].items():
    print(f"AUROC({f}) = {r['none']:.3f}, after O {r['organ']:.3f}, after control {r['control']:.3f}, n1 = {r['n1']}")
if "detection_after_erasure_mean" in out:
    m = out["detection_after_erasure_mean"]
    print(f"AUROC mean over {m['n']} findings with n1 >= 5: {m['none']:.3f}, after O {m['organ']:.3f}, after control {m['control']:.3f}; O below control for {m['organ_below_control']}")
if a.out:
    json.dump(out, open(a.out, "w"), indent=1, default=float)
