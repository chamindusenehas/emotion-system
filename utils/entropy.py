import numpy as np

def confidence_from_probs(probs):
    eps = 1e-8
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))
    confidence = 1 - (entropy / max_entropy)
    return confidence
