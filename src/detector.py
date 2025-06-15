import torch
import numpy as np
from ultralytics import YOLO

class YOLOv11Detector:
    """
    Loads a YOLOv11 model and runs inference on images/frames.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Runs detection on a single image/frame.
        Returns: list of dicts with keys: 'bbox', 'conf', 'cls', 'label'
        """
        results = self.model(image)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, clss):
                detections.append({
                    'bbox': box,  # [x1, y1, x2, y2]
                    'conf': conf,
                    'cls': int(cls),
                    'label': self.model.names[int(cls)]
                })
        return detections 