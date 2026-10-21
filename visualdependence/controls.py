"""Control construction: K rigid copies of O at its relative eccentricity, or None (N/A) if fewer than K fit."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes


def placement_body(body, plane=(0, 1)):
    """The body with interior air filled, slice by slice."""
    body = np.asarray(body, bool)
    if body.all():
        return body
    others = [ax for ax in range(body.ndim) if ax not in plane]
    if not others:
        return binary_fill_holes(body)
    out = np.zeros_like(body)
    for i in range(body.shape[others[0]]):
        sl = [slice(None)] * body.ndim; sl[others[0]] = i
        out[tuple(sl)] = binary_fill_holes(body[tuple(sl)])
    return out


def body_extent(pbody, c_body, plane=(0, 1), n=720, step=0.5):
    """R_B(u): the distance from c_B along u to the point where the ray first leaves the outline B,
    B the body projected onto the slice plane. Returns (angles, R)."""
    a0, a1 = plane
    others = tuple(ax for ax in range(pbody.ndim) if ax not in plane)
    proj = pbody.any(axis=others) if others else pbody
    if a0 > a1:
        proj = proj.T
    H, W = proj.shape
    c = np.array([c_body[a0], c_body[a1]], float)
    angs = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ts = np.arange(0, float(np.hypot(H, W)) + step, step)
    u = np.stack([np.cos(angs), np.sin(angs)], 1)
    ij = np.rint(c[None, None, :] + ts[None, :, None] * u[:, None, :]).astype(int)
    inside = (ij[..., 0] >= 0) & (ij[..., 0] < H) & (ij[..., 1] >= 0) & (ij[..., 1] < W)
    ok = np.zeros(inside.shape, bool)
    ok[inside] = proj[ij[..., 0][inside], ij[..., 1][inside]]
    first_out = np.where(ok.all(1), len(ts), np.argmin(ok, axis=1))
    R = ts[np.clip(first_out - 1, 0, len(ts) - 1)]
    R[first_out == 0] = 0.0
    return angs, R


def rigid_controls(mask, body, k=2, dilate=1, plane=(0, 1), exclude=None, c_body=None, place_body=None, n_angles=72):
    """K controls, each a copy of O rotated by a quarter turn and centered in-plane at c_B + rho_O R_B(u) u,
    on O's slice or on a slice shifted by h, -h, 2h, -2h, ... (h = O's axial extent plus two voxels).
    A candidate is accepted if every voxel lies inside the body, with interior air filled, and none comes within
    one voxel of O, of any other finding present in the case, or of a control already placed. Candidates are
    visited by slice, then rotation, then direction, turning away from u_j by 5 degrees at a time.
    Returns K boolean masks, or None."""
    mask = np.asarray(mask, bool); body = np.asarray(body, bool)
    pbody = np.asarray(place_body, bool) if place_body is not None else placement_body(body, plane)
    a0, a1 = plane
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    shape = np.array(mask.shape)
    body_all = bool(pbody.all())
    c_body = np.argwhere(body).mean(0) if c_body is None else np.asarray(c_body, float)
    c_O = coords.mean(0)
    d = c_O - c_body
    theta = float(np.arctan2(d[a1], d[a0]))
    excl = binary_dilation(mask, iterations=dilate) if dilate else mask.copy()
    if exclude is not None:
        ex = np.asarray(exclude, bool)
        excl = excl | (binary_dilation(ex, iterations=dilate) if dilate else ex)
    e = np.argwhere(excl); ex_lo, ex_hi = e.min(0), e.max(0)
    rel = coords - c_O

    def rotate(q):                                   # quarter turn q about the centroid
        r = rel.copy(); x, y = rel[:, a0].copy(), rel[:, a1].copy()
        if q == 1:
            r[:, a0], r[:, a1] = -y, x
        elif q == 2:
            r[:, a0], r[:, a1] = -x, -y
        elif q == 3:
            r[:, a0], r[:, a1] = y, -x
        return r
    rels = [(r, r.min(0), r.max(0)) for r in (rotate(q) for q in range(4))]
    stride = max(1, len(coords) // 2000)

    def place(target, q):
        relq, rlo, rhi = rels[q]
        lo = np.rint(rlo + target).astype(int); hi = np.rint(rhi + target).astype(int)
        if (lo < 0).any() or (hi >= shape).any():
            return None
        clear = (hi < ex_lo).any() or (lo > ex_hi).any()
        if stride > 1:                               # quick rejection on a subsample
            pr = tuple(np.rint(relq[::stride] + target).astype(int).T)
            if (not body_all and not pbody[pr].all()) or (not clear and excl[pr].any()):
                return None
        idx = tuple(np.rint(relq + target).astype(int).T)
        if (body_all or pbody[idx].all()) and (clear or not excl[idx].any()):
            m = np.zeros(mask.shape, bool); m[idx] = True
            return m
        return None

    angs_b, R_b = body_extent(pbody, c_body, plane)
    R_at = lambda a: float(R_b[int(np.rint((a % (2 * np.pi)) / (2 * np.pi) * len(angs_b))) % len(angs_b)])
    rho_O = float(np.hypot(d[a0], d[a1])) / max(R_at(theta), 1e-9)
    if rho_O > 1.0:                                  # a mask whose centroid lies outside B has no control
        return None
    others = [ax for ax in range(mask.ndim) if ax not in plane]
    levels = [0.0]
    if others:
        az = others[0]
        h = float(rel[:, az].max() - rel[:, az].min() + 1 + 2 * dilate)
        for i in range(1, int(shape[az] // max(h, 1.0)) + 2):
            levels += [i * h, -i * h]
    step = 2 * np.pi / n_angles
    out = []
    for j in range(1, k + 1):
        u_j = theta + 2 * np.pi * j / (k + 1)
        found = None
        for dz in levels:
            for q in range(4):
                for i in range(n_angles):
                    a = u_j + ((i + 1) // 2) * step * (1 if i % 2 == 0 else -1)
                    target = c_O.copy()
                    target[a0] = c_body[a0] + rho_O * R_at(a) * np.cos(a)
                    target[a1] = c_body[a1] + rho_O * R_at(a) * np.sin(a)
                    if others:
                        target[az] = c_O[az] + dz
                    found = place(target, q)
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            return None
        out.append(found)
        excl = excl | (binary_dilation(found, iterations=dilate) if dilate else found)
        f = np.argwhere(found); ex_lo = np.minimum(ex_lo, f.min(0)); ex_hi = np.maximum(ex_hi, f.max(0))
    return out
