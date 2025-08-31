import time

class FPSCalculator:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.fps_values = []
        self.prev_time = 0

    def update(self):
        """
        Updates the FPS calculation and returns the current average FPS.

        Returns:
            float: The current average FPS
        """
        current_time = time.time()

        if self.prev_time > 0:
            fps = 1 / (current_time - self.prev_time)
            self.fps_values.append(fps)

            if len(self.fps_values) > self.buffer_size:
                self.fps_values.pop(0)

            avg_fps = sum(self.fps_values) / len(self.fps_values)
        else:
            avg_fps = 0

        self.prev_time = current_time
        return avg_fps

    def reset(self):
        """Reset the FPS calculator."""
        self.fps_values = []
        self.prev_time = 0