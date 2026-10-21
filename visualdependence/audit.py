"""The audit: erase each region and its K controls, then the four measures."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from . import measures as M
from . import perturb, controls


@dataclass
class Case:
    """An image x with, for each finding f, a binary label y_f and the mask O it maps to."""
    image: np.ndarray
    masks: dict                                   # finding -> O
    y: dict                                       # finding -> y_f
    body: np.ndarray                              # body mask; its mean intensity is the fill value
    region_of: dict = field(default_factory=dict) # finding -> id of O; findings with one id form F_O
    class_label: int | None = None                # diagnostic label y (label-specificity), if any
    exclude: np.ndarray | None = None             # further regions a control must not touch

    def __post_init__(self):
        for f in self.masks:
            self.region_of.setdefault(f, f)


@dataclass
class Pair:
    """A case-finding pair: s_f(x), d_organ(x, f) and the K control drops d(x, c_k, f)."""
    case: object
    finding: str
    region: str
    y: int
    s: float
    d_organ: float
    d_control: np.ndarray
    class_label: int | None = None


def other_findings(case: Case, f: str):
    """Every other finding present in the case, plus case.exclude."""
    m = np.zeros(np.asarray(case.image).shape, bool)
    for g, gm in case.masks.items():
        if g != f and int(case.y.get(g, 0)) == 1:
            m |= np.asarray(gm, bool)
    if case.exclude is not None:
        m |= np.asarray(case.exclude, bool)
    return m if m.any() else None


def case_controls(case: Case, f: str, K=2, plane=(0, 1), pbody=None, c_body=None):
    return controls.rigid_controls(case.masks[f], case.body, k=K, dilate=1, plane=plane, exclude=other_findings(case, f),
                                   c_body=c_body, place_body=pbody)


def drops(adapter, case: Case, K=2, plane=(0, 1), case_id=None, findings=None):
    """The drops of every finding of one case. Returns (pairs, N/A findings)."""
    fill = perturb.body_mean(case.image, case.body)
    pbody = controls.placement_body(case.body, plane)
    c_body = np.argwhere(case.body).mean(0)
    phi = adapter.embed_image(case.image)
    pairs, na = [], []
    for f, O in case.masks.items():
        if findings is not None and f not in findings:
            continue
        c = case_controls(case, f, K, plane, pbody, c_body)
        if c is None:                             # constructibility: fewer than K copies fit
            na.append(f)
            continue
        s = adapter.score(phi, f)
        d_organ = s - adapter.score_image(perturb.erase(case.image, O, fill), f)
        d_control = np.array([s - adapter.score_image(perturb.erase(case.image, ck, fill), f) for ck in c])
        pairs.append(Pair(case_id, f, case.region_of[f], int(case.y[f]), float(s), float(d_organ), d_control, case.class_label))
    return pairs, na


def std_null(pairs) -> float:
    """One std_null over all case-finding pairs, present and absent."""
    return M.std_null(np.stack([p.d_control for p in pairs]))


def dependence(pairs, std_null):
    """occ_bar over present pairs, with a 95% bootstrap interval."""
    present = [p for p in pairs if p.y == 1]
    o = M.occ(np.array([p.d_organ for p in present]), np.stack([p.d_control for p in present]), std_null)
    m, lo, hi = M.bootstrap_mean_ci(o)
    return dict(occ_bar=m, ci=(lo, hi), n=len(present))


def finding_specificity(pairs, std_null, min_cases=5):
    """R_O for each mask O shared by two or more findings, pooled over cases; reported only where it rests on at
    least `min_cases` cases."""
    out = {}
    for region in sorted({p.region for p in pairs}):
        rp = [p for p in pairs if p.region == region]
        if len({p.finding for p in rp}) < 2:
            continue
        by_case = {}
        for p in rp:
            by_case.setdefault(p.case, []).append(p)
        terms = []
        for cp in by_case.values():
            if len(cp) < 2:
                continue
            o = M.occ(np.array([p.d_organ for p in cp]), np.stack([p.d_control for p in cp]), std_null)
            on = np.stack([M.occ_null(p.d_control, std_null) for p in cp])
            terms.append(M.R_O_terms(o, on))
        if len(terms) >= max(min_cases, 2):
            r, lo, hi = M.bootstrap_ratio_ci(terms)
            out[region] = dict(R_O=r, ci=(lo, hi), n=len(terms))
        else:
            out[region] = dict(R_O=None, ci=None, n=len(terms))
    return out


def label_specificity(pairs, std_null):
    """For each finding: occ_bar(y=1), occ_bar(y=0), Delta_f and G_f. With a diagnostic label the groups are the
    classes; with a presence label y=1 is y_f=1 and y=0 is y_{F_O}=0. G_f is established where the interval of
    Delta_f lies above 0."""
    occ_of = {(p.case, p.finding): float(M.occ(np.array([p.d_organ]), p.d_control[None, :], std_null)[0]) for p in pairs}

    def record(g1, g0):
        o1 = [occ_of[(p.case, p.finding)] for p in g1]; o0 = [occ_of[(p.case, p.finding)] for p in g0]
        s1 = [p.s for p in g1]; s0 = [p.s for p in g0]
        d, dlo, dhi = (v / std_null for v in M.bootstrap_diff_ci(s1, s0))
        return dict(n1=len(g1), n0=len(g0), occ_y1=M.bootstrap_mean_ci(o1), occ_y0=M.bootstrap_mean_ci(o0),
                    gap=M.bootstrap_diff_ci(o1, o0), Delta_f=(d, dlo, dhi),
                    G_f=M.bootstrap_G_f_ci(o1, o0, s1, s0, std_null), established=bool(dlo > 0))

    out = {}
    if any(p.class_label is not None for p in pairs):                  # diagnostic label
        for f in sorted({p.finding for p in pairs}):
            fp = [p for p in pairs if p.finding == f and p.class_label is not None]
            g1 = [p for p in fp if p.class_label == 1]; g0 = [p for p in fp if p.class_label == 0]
            if g1 and g0:
                out[f] = record(g1, g0)
        return out
    for region in sorted({p.region for p in pairs}):                    # presence label
        rp = [p for p in pairs if p.region == region]
        n_present = {}
        for p in rp:
            n_present[p.case] = n_present.get(p.case, 0) + p.y
        absent = {c for c, n in n_present.items() if n == 0}            # y_{F_O} = 0
        for f in sorted({p.finding for p in rp}):
            fp = [p for p in rp if p.finding == f]
            g1 = [p for p in fp if p.y == 1]; g0 = [p for p in fp if p.case in absent]
            if g1 and g0:
                out[f] = record(g1, g0)
    return out


def position_terms(adapter, case: Case, offsets, axis, K=2, plane=(0, 1), guard=None, blank_distance=None):
    """pos terms of every present finding of one case: the image and the mask are translated together by each
    offset delta and the occlusion difference is recomputed. A pair is skipped if a shift fails the
    on-manifold guard, and N/A if its controls do not fit at some offset. Returns (terms, n_skipped, n_na)."""
    fill = perturb.body_mean(case.image, case.body)
    pbody = controls.placement_body(case.body, plane)
    terms, skipped, na = [], 0, 0
    for f, O in case.masks.items():
        if int(case.y.get(f, 0)) != 1:
            continue
        ex = other_findings(case, f)
        occ_d, null_d, ok = [], [], True
        for delta in offsets:
            x_d = perturb.translate(case.image, delta, axis, fill)
            O_d = perturb.translate_mask(O, delta, axis)
            phi = adapter.embed_image(x_d)
            if not O_d.any() or (guard is not None and not guard.passes(phi, blank_distance)):
                ok = "skip"; break
            c = controls.rigid_controls(O_d, case.body, k=K, dilate=1, plane=plane,
                                        exclude=perturb.translate_mask(ex, delta, axis) if ex is not None else None, place_body=pbody)
            if c is None:
                ok = "na"; break
            s = adapter.score(phi, f)
            d_organ = s - adapter.score_image(perturb.erase(x_d, O_d, fill), f)
            d_control = np.array([s - adapter.score_image(perturb.erase(x_d, ck, fill), f) for ck in c])
            occ_d.append(d_organ - d_control.mean())          # std_null cancels in pos
            null_d.append(M.null(d_control))
        if ok == "na":
            na += 1
        elif ok == "skip" or len(occ_d) < 3:
            skipped += 1
        else:
            terms.append(M.pos_terms(np.array(occ_d), np.stack(null_d)))
    return terms, skipped, na


def position(terms, n_boot: int = 2000, seed: int = 0):
    """pos pooled over case-finding pairs, with a 95% bootstrap interval."""
    terms = list(terms)
    if not terms:
        return dict(pos=float("nan"), ci=(float("nan"), float("nan")), n=0)
    rng = np.random.default_rng(seed)
    boots = [M.pooled_ratio([terms[i] for i in rng.integers(0, len(terms), len(terms))]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(pos=M.pooled_ratio(terms), ci=(float(lo), float(hi)), n=len(terms))
