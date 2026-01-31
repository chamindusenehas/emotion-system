import numpy as np

class ConfidenceAwareFusion:
    def __init__(self):
        self.eps = 1e-6

    def fuse(self, V, A, Cv, Ca):
        wv = Cv / (Cv + Ca + self.eps)
        wa = Ca / (Cv + Ca + self.eps)
        fused = wv * V + wa * A
        return fused, wv, wa
