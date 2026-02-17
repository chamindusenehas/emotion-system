import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

class CameraStream:
    def __init__(self, device="cpu", detect_every=10):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.mtcnn = MTCNN(keep_all=False, device=device)
        self.detect_every = detect_every

        self.frame_count = 0
        self.last_box = None

    def get_face(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, frame, None

        self.frame_count += 1

        if self.frame_count % self.detect_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            boxes, _ = self.mtcnn.detect(img)

            if boxes is not None:
                self.last_box = boxes[0].astype(int)
            else:
                self.last_box = None

        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return None, frame, self.last_box

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)

            return face, frame, self.last_box

        return None, frame, None
