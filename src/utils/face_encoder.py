import os
import face_recognition

class FaceEncoder:
    def __init__(self):
        self.encodings = {}

    def load_face_encoding(self, name, image_path):
        """
        Load a face encoding from an image file.

        Args:
            name (str): Name to associate with the face
            image_path (str): Path to the image file

        Returns:
            bool: True if encoding was successfully loaded, False otherwise
        """
        if not os.path.exists(image_path):
            print(f"Warning: Face image not found at {image_path}")
            return False

        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                self.encodings[name] = face_encodings[0]
                print(f"Loaded face encoding for {name}")
                return True
            else:
                print(f"Warning: No face found in {image_path}")
                return False
        except Exception as e:
            print(f"Error loading face encoding for {name}: {e}")
            return False

    def get_encoding(self, name):
        """Get face encoding by name."""
        return self.encodings.get(name)

    def has_encoding(self, name):
        """Check if encoding exists for name."""
        return name in self.encodings