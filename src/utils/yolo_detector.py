import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="models/yolo11s.pt", confidence=0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.results = []

    def detect(self, frame, enabled_classes=None, detection_interval=5, frame_count=0):
        """
        Perform YOLO detection on the frame.

        Args:
            frame: Input frame
            enabled_classes: List of enabled class IDs
            detection_interval: Run detection every N frames
            frame_count: Current frame number

        Returns:
            list: YOLO detection results
        """
        if frame_count % detection_interval == 0:
            if enabled_classes:
                self.results = self.model.predict(frame, conf=self.confidence, verbose=False, classes=enabled_classes)
            else:
                self.results = self.model.predict(frame, conf=self.confidence, verbose=False)

        return self.results

    def draw_detections(self, frame, results, class_info):
        """
        Draw YOLO detection boxes and labels on the frame.

        Args:
            frame: Frame to draw on
            results: YOLO detection results
            class_info: Dictionary mapping class IDs to names and colors

        Returns:
            frame: Frame with drawn detections
        """
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0]

                if cls in class_info:
                    label = class_info[cls]['name']
                    color = class_info[cls]['color']
                else:
                    label = self.model.names[cls]
                    color = (255, 0, 0) if label == "person" else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame