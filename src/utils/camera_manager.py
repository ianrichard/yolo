import cv2
import numpy as np
from typing import Tuple, Optional

class CameraManager:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def initialize(self) -> bool:
        """Initialize the camera with specified settings."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False

        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return False, None

        return ret, frame

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release the camera and cleanup."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()