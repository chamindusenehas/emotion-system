import numpy as np

def confidence_from_entropy(probs):
    eps = 1e-8
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))
    return 1 - (entropy / max_entropy)

def is_speech(audio, threshold=0.01):
    rms = np.sqrt(np.mean(audio**2))
    print(f"Audio RMS: {rms:.6f}")
    return rms > threshold