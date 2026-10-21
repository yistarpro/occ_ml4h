"""Wrap a new dual-encoder VLM: implement embed_image and score, then build one Case per image."""
import numpy as np
from visualdependence import VLMAdapter, Case


class MyAdapter(VLMAdapter):
    def __init__(self, model, text_vectors):
        self.model, self.psi = model, text_vectors          # psi[f]: the unit text embedding of finding f

    def embed_image(self, image):
        z = self.model.encode_image(image)                   # the model's own preprocessing and forward pass
        return z / np.linalg.norm(z)                         # phi(x)

    def score(self, embedding, finding):
        return float(embedding @ self.psi[finding])          # s_f(x) = <phi(x), psi_f>


def my_case(image, organ_mask, body_mask, label):
    return Case(image=image, masks={"finding": organ_mask}, y={"finding": label}, body=body_mask)
