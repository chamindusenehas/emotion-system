import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

class VisualEmotionModel:
    def __init__(self, device="cpu"):
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(
            "dima806/facial_emotions_image_detection"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "dima806/facial_emotions_image_detection"
        ).to(self.device)

        self.model.eval()
        self.labels = self.model.config.id2label

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        return probs.squeeze().cpu().numpy(), self.labels
