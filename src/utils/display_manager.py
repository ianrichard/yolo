import cv2

class DisplayManager:
    def __init__(self, window_name="YOLO11 + Face Recognition"):
        self.window_name = window_name

    def draw_fps(self, frame, fps):
        """Draw FPS counter on frame."""
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def show_frame(self, frame):
        """Display the frame in the window."""
        cv2.imshow(self.window_name, frame)

    def check_key(self):
        """
        Check for keyboard input.

        Returns:
            str: The pressed key ('q' for quit, None for no key pressed)
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'q'
        return None

    def cleanup(self):
        """Cleanup display resources."""
        cv2.destroyAllWindows()