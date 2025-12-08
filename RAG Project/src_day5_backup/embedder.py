from typing import List
import numpy as np

class BaseEmbedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class DummyEmbedder(BaseEmbedder):
    def embed(self, texts: List[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), 8)).astype('float32')
