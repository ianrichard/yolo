import cv2

class CameraManager:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def initialize(self):
        """
        Initialize the camera with specified settings.

        Returns:
            bool: True if camera was successfully initialized, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False

        return True

    def read_frame(self):
        """
        Read a frame from the camera.

        Returns:
            tuple: (success, frame) where success is bool and frame is the captured image
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")

        return ret, frame

    def is_opened(self):
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Release the camera and cleanup."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()