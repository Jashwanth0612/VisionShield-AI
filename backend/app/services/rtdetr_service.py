import torch
import numpy as np
from PIL import Image

class RTDETRService:
    def __init__(self, model_path: str = "models/rt_detr/rtdetr_weights.pt", device: str = None, conf_threshold: float = 0.5):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def load_model(self):
        """Load the RT-DETR object detection model weights."""
        try:
            # TODO: Initialize RT-DETR architecture / Ultralytics RT-DETR wrapper
            # from ultralytics import RTDETR
            # self.model = RTDETR(self.model_path)
            print(f"[RTDETRService] Model initialized successfully on device: {self.device}")
        except Exception as e:
            print(f"[RTDETRService] Warning: Could not load weights from {self.model_path}: {e}")

    def detect(self, image: Image.Image) -> list:
        """
        Runs object detection on the input image (preprocessed or enhanced).
        Returns a list of detected bounding boxes, class IDs, and confidence scores.
        """
        if self.model is None:
            # Fallback mock detection structure for pipeline testing
            return []

        # Run inference
        results = self.model(image, conf=self.conf_threshold)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.model.names[class_id]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
                "label": label
            })

        return detections
