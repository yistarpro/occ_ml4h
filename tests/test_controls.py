import numpy as np
from scipy.ndimage import binary_dilation
from visualdependence.controls import rigid_controls, placement_body


def _disk(shape, c, r):
    i, j = np.indices(shape)
    return (i - c[0]) ** 2 + (j - c[1]) ** 2 <= r * r


def test_central_region_has_no_control_in_2d():
    assert rigid_controls(_disk((128, 128), (64, 64), 10), _disk((128, 128), (64, 64), 50), k=2) is None


def test_controls_are_disjoint_copies_inside_the_body():
    body = _disk((128, 128), (64, 64), 50)
    O = _disk((128, 128), (64, 92), 8) | _disk((128, 128), (70, 97), 4)
    c = rigid_controls(O, body, k=2)
    assert c is not None and len(c) == 2
    placed = binary_dilation(O)
    for m in c:
        assert m.sum() == O.sum() and placement_body(body)[m].all() and not (placed & m).any()
        placed |= binary_dilation(m)
    assert rigid_controls(O, body, k=2)[0].tobytes() == c[0].tobytes()      # no random element


def test_a_central_region_moves_to_other_slices_in_3d():
    body = np.broadcast_to(_disk((128, 128), (64, 64), 50), (40, 128, 128)).copy()
    O = np.zeros_like(body); O[17:23] = _disk((128, 128), (64, 64), 8)
    c = rigid_controls(O, body, k=2, plane=(1, 2))
    assert c is not None and all(not (m & O).any() for m in c)
    O[:] = False; O[2:38] = _disk((128, 128), (64, 64), 8)          # spans the volume: no room
    assert rigid_controls(O, body, k=2, plane=(1, 2)) is None
