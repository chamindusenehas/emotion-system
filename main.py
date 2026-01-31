from models.visual_model import VisualEmotionModel
from models.audio_model import AudioEmotionModel
from models.fusion import ConfidenceAwareFusion
from utils.entropy import confidence_from_probs
from utils.smoothing import EMA

visual_ema = EMA(alpha=0.3)
audio_ema = EMA(alpha=0.3)

visual_model = VisualEmotionModel("path to visual_model")
audio_model = AudioEmotionModel("path to audio_model")

fusion = ConfidenceAwareFusion()

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'] # Example labels

while True:
    # 1. Capture frame & audio window
    face_tensor = "" # get_face_tensor
    audio_features = "" # get_audio_features

    # 2. Predict emotions
    V = visual_model.predict(face_tensor)
    A = audio_model.predict(audio_features)

    # 3. Confidence estimation
    Cv = confidence_from_probs(V)
    Ca = confidence_from_probs(A)

    # 4. Temporal smoothing
    Cv_s = visual_ema.update(Cv)
    Ca_s = audio_ema.update(Ca)

    # 5. Fusion
    P, wv, wa = fusion.fuse(V, A, Cv_s, Ca_s)

    emotion = EMOTION_LABELS[P.argmax()]
    print(emotion, wv, wa)
