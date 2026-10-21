"""The model interface: a dual-encoder VLM that scores an image x against a finding f."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class VLMAdapter(ABC):
    @abstractmethod
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """The unit image embedding phi(x), including the model's own preprocessing."""

    @abstractmethod
    def score(self, embedding: np.ndarray, finding: str) -> float:
        """The finding score s_f(x) from phi(x)."""

    def score_image(self, image: np.ndarray, finding: str) -> float:
        return self.score(self.embed_image(image), finding)
