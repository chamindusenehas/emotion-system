import numpy as np

def fuse_emotions(V_probs, A_probs, V_conf, A_conf):
    total_conf = V_conf + A_conf
    if total_conf == 0:
        weights = [0.5, 0.5]
    else:
        weights = [V_conf / total_conf, A_conf / total_conf]

    fused_probs = V_probs * weights[0] + A_probs * weights[1]
    fused_emotion_id = np.argmax(fused_probs)
    return fused_emotion_id, fused_probs

def fuse_emotions_aud(V_probs, A_probs_mapped, V_conf, A_conf):
    total_conf = V_conf + A_conf
    if total_conf == 0:
        weights = [0.5, 0.5]
    else:
        weights = [V_conf / total_conf, A_conf / total_conf]
    fused_probs = V_probs * weights[0] + A_probs_mapped * weights[1]
    fused_id = np.argmax(fused_probs)
    return fused_id, fused_probs
