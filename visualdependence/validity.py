"""On-manifold guard. Containment is decided per finding in each adapter; constructibility in controls."""
from __future__ import annotations
import numpy as np


class OnManifoldGuard:
    """Fits a Gaussian to the embeddings of unperturbed images and scores a perturbed image by its Mahalanobis
    distance from that fit, in units of the spread of the real images' own distances (Ledoit-Wolf covariance).
    A perturbation passes if its distance is less than half the blank's."""

    def fit(self, real_embeddings: np.ndarray) -> "OnManifoldGuard":
        from sklearn.covariance import LedoitWolf
        X = np.asarray(real_embeddings, float)
        lw = LedoitWolf().fit(X)
        self.mu_, self.prec_ = lw.location_, lw.precision_
        m = np.array([self._mahalanobis(x) for x in X])
        self.m0_, self.ms_ = float(m.mean()), float(m.std() + 1e-12)
        return self

    def _mahalanobis(self, x) -> float:
        d = np.asarray(x, float) - self.mu_
        return float(np.sqrt(d @ self.prec_ @ d))

    def distance(self, embedding) -> float:
        """Mahalanobis distance from the real images, in units of their spread (0 = a typical real image)."""
        return (self._mahalanobis(embedding) - self.m0_) / self.ms_

    def passes(self, embedding, blank_distance: float, frac: float = 0.5) -> bool:
        return self.distance(embedding) < frac * blank_distance

    def save(self, path: str, blank_distance: float):
        np.savez(path, mu=self.mu_, prec=self.prec_, m0=self.m0_, ms=self.ms_, blank_distance=blank_distance)

    @classmethod
    def load(cls, path: str):
        z = np.load(path)
        g = cls(); g.mu_, g.prec_, g.m0_, g.ms_ = z["mu"], z["prec"], float(z["m0"]), float(z["ms"])
        return g, float(z["blank_distance"])
