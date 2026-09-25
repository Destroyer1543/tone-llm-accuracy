import numpy as np
from huggingface_hub import hf_hub_download


class JumpReLUSAE:
    def __init__(self, repo, filename):
        path = hf_hub_download(repo, filename)
        p = np.load(path)
        self.W_enc = p["W_enc"].astype(np.float32)
        self.b_enc = p["b_enc"].astype(np.float32)
        self.W_dec = p["W_dec"].astype(np.float32)
        self.b_dec = p["b_dec"].astype(np.float32)
        self.threshold = p["threshold"].astype(np.float32)
        self.d_model, self.d_sae = self.W_enc.shape

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        acts = np.where(pre > self.threshold, pre, 0.0)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def feature_dirs(self, idxs):
        return self.W_dec[idxs]
