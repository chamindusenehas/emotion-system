import numpy as np

def fuse_emotions(V_probs, A_probs, V_conf, A_conf):
    
    if V_conf > 0.75 and A_conf < 0.4:
        A_conf *= 0.3
    
    V_w = V_conf ** 2
    A_w = A_conf ** 2

    total = V_w + A_w
    if total == 0:
        return np.argmax(V_probs), V_probs

    V_w /= total
    A_w /= total

    fused_probs = V_probs * V_w + A_probs * A_w
    fused_probs /= fused_probs.sum() + 1e-8

    fused_id = np.argmax(fused_probs)
    return fused_id, fused_probs