import numpy as np
from visualdependence import measures as M


def test_null_is_mean_zero():
    assert abs(M.null(np.array([[1.0, 2.0, 3.0, 4.0]])).sum()) < 1e-9


def test_null_variance_matches_the_occ_contrast():
    rng = np.random.default_rng(0)
    for K in (2, 4):
        d = rng.normal(0, 1, (200000, K + 1))            # column 0 plays O, the rest the controls
        assert abs(M.null(d[:, 1:]).std() / (d[:, 0] - d[:, 1:].mean(1)).std() - 1) < 0.01


def test_occ():
    assert abs(M.occ(np.array([1.0]), np.zeros((1, 2)), 0.5)[0] - 2.0) < 1e-9
    assert abs(M.occ(np.array([0.3]), np.full((1, 2), 0.3), 0.5)[0]) < 1e-9


def test_R_O_above_1_when_occ_separates_findings():
    n, d = M.R_O_terms(np.array([3.0, -3.0]), np.array([[0.01, -0.01], [-0.01, 0.01]]))
    assert n / d > 5


def test_pos_below_1_when_occ_is_stable_under_shifts():
    rng = np.random.default_rng(0)
    n, d = M.pos_terms(np.array([2.0, 2.01, 1.99, 2.0]), rng.normal(0, 1, (4, 2)))
    assert n / d < 1


def test_G_f_is_the_gap_over_the_separation():
    g, lo, hi = M.bootstrap_G_f_ci([2.0, 2.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], std_null=0.5)
    assert abs(g - 1.0) < 1e-9                           # gap 2 / Delta_f (1 / 0.5)
