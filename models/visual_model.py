import torch
import torch.nn.functional as F

class VisualEmotionModel:
    def __init__(self, model):
        self.model = model.eval()

    def predict(self, face_tensor):
        with torch.no_grad():
            logits = self.model(face_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.squeeze().cpu().numpy()
