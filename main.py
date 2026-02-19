from models.visual_model import VisualEmotionModel
from streaming.camera import CameraStream
from streaming.audio_stream import AudioStream
from models.audio_model import AudioEmotionModel
from models.fusion_model import fuse_emotions
import numpy as np
import cv2
from collections import deque
import threading
import sounddevice as sd
from utils.entrophy import confidence_from_entropy, is_speech   


# Initialize models and streams
visual_model = VisualEmotionModel()
camera = CameraStream(detect_every=10)
V_labels = visual_model.labels
audio_model = AudioEmotionModel()

A_sample_rate = 16000
A_duration = 2 
audio_buffer = None
audio_stream = AudioStream(duration=2) 

SMOOTH_FRAMES = 5
emotion_history = deque(maxlen=SMOOTH_FRAMES)

def record_audio_loop():
    global audio_buffer
    while True:
        audio = sd.rec(int(A_sample_rate * A_duration), samplerate=A_sample_rate, channels=1)
        sd.wait()
        audio_buffer = np.squeeze(audio)


audio_thread = threading.Thread(target=record_audio_loop, daemon=True)
audio_thread.start()

audio_to_visual_map = {
    "angry": "angry",
    "calm": "neutral",
    "disgust": "disgust",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": None 
}


# Map audio emotions to visual emotions
def map_audio_to_visual_probs(audio_probs, audio_labels, visual_labels):
    mapped = np.zeros(len(visual_labels))

    for i, a_label in enumerate(audio_labels):
        v_label = audio_to_visual_map.get(a_label)
        if v_label is None:
            continue
        if v_label in visual_labels:
            v_idx = visual_labels.index(v_label)
            mapped[v_idx] += audio_probs[i]

    if mapped.sum() > 0:
        mapped /= mapped.sum()

    return mapped


while True:
    # Visual processing
    face, frame, box = camera.get_face()

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    if face is not None:
        V_probs, _ = visual_model.predict(face)
        V_id = np.argmax(V_probs)
        V_conf = V_probs[V_id]
    else:
        V_probs = np.zeros(len(V_labels))
        V_conf = 0


    # Audio processing
    if audio_buffer is not None and is_speech(audio_buffer):
        A_probs, A_labels = audio_model.predict(audio_buffer)
        A_probs_mapped = map_audio_to_visual_probs(A_probs, A_labels, V_labels)

        NEUTRAL_IDX = next(k for k, v in V_labels.items() if v == "neutral")
        if A_probs_mapped.sum() > 0:
            A_probs_mapped[NEUTRAL_IDX] += 0.15
            A_probs_mapped /= A_probs_mapped.sum()

        A_conf = confidence_from_entropy(A_probs_mapped)

    else:
        A_probs = np.zeros(len(V_labels))
        A_labels = V_labels
        A_probs_mapped = np.zeros(len(V_labels))
        A_conf = 0.0

    # Fusion
    fused_id, fused_probs = fuse_emotions(V_probs, A_probs_mapped, V_conf, A_conf)
    fused_emotion = V_labels[fused_id]

    # Smoothing
    emotion_history.append(fused_emotion)
    smooth_emotion = max(set(emotion_history), key=emotion_history.count)
    smooth_confidence = fused_probs[fused_id]

    # Visual top emotion
    V_top_id = np.argmax(V_probs)
    V_top_emotion = V_labels[V_top_id]
    V_top_conf = V_probs[V_top_id]

    # Audio top emotion
    if A_conf > 0:
        A_top_id = np.argmax(A_probs)
        A_top_emotion = A_labels[A_top_id]
        A_top_conf = A_probs[A_top_id]
    else:
        A_top_emotion = "N/A"
        A_top_conf = 0.0

    print(
        f"Visual: {V_top_emotion} ({V_top_conf:.2f}), "
        f"Audio: {A_top_emotion} ({A_top_conf:.2f}), "
        f"Fused: {smooth_emotion} ({smooth_confidence:.2f})"
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()