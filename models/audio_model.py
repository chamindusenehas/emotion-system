import torch
import torch.nn.functional as F

class AudioEmotionModel:
    def __init__(self, model):
        self.model = model.eval()

    def predict(self, audio_features):
        with torch.no_grad():
            logits = self.model(audio_features)
            probs = F.softmax(logits, dim=1)
        return probs.squeeze().cpu().numpy()
