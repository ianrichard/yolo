import os
import numpy as np
from typing import Optional

# Suppress TensorFlow/MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.config_loader import load_config
from utils.class_handler import get_enabled_classes
from utils.face_recognition_utils import recognize_faces
from utils.fps_calculator import FPSCalculator
from utils.yolo_detector import YOLODetector
from utils.camera_manager import CameraManager
from utils.face_loader import FaceLoader
from utils.display_manager import DisplayManager

def run() -> None:
    camera: Optional[CameraManager] = None
    display: Optional[DisplayManager] = None

    try:
        # Initialize components
        camera = CameraManager(camera_index=0, width=640, height=480)
        yolo_detector = YOLODetector("models/yolo11s.pt")
        face_loader = FaceLoader("images")
        display = DisplayManager()

        config = load_config()
        enabled_classes, class_info = get_enabled_classes(config)

        # Load all face encodings automatically
        known_face_encodings = face_loader.load_all_faces()

        if not camera.initialize():
            return

        fps_calc = FPSCalculator()
        frame_count = 0

        while camera.is_opened():
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                break

            frame_count += 1
            avg_fps = fps_calc.update()

            display_frame = frame.copy()

            # Face recognition (now works with multiple people)
            display_frame = recognize_faces(frame, known_face_encodings, frame_count)

            # YOLO detection
            yolo_results = yolo_detector.detect(frame, enabled_classes, detection_interval=5, frame_count=frame_count)
            display_frame = yolo_detector.draw_detections(display_frame, yolo_results, class_info)

            # Display
            display_frame = display.draw_fps(display_frame, avg_fps)
            display.show_frame(display_frame)

            # Check for exit
            if display.check_key() == 'q':
                break

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if camera is not None:
            camera.release()
        if display is not None:
            display.cleanup()

if __name__ == "__main__":
    run()