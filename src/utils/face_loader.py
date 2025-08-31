import os
import glob
import face_recognition

class FaceLoader:
    def __init__(self, images_directory="images"):
        self.images_directory = images_directory
        self.encodings = {}
        self.names = []

    def load_all_faces(self):
        """
        Automatically load all .jpg images from the images directory.
        Uses the filename (without extension) as the person's name.

        Returns:
            dict: Dictionary mapping names to face encodings
        """
        if not os.path.exists(self.images_directory):
            print(f"Warning: Images directory '{self.images_directory}' not found")
            return {}

        # Find all .jpg files in the directory
        jpg_pattern = os.path.join(self.images_directory, "*.jpg")
        jpg_files = glob.glob(jpg_pattern)

        if not jpg_files:
            print(f"Warning: No .jpg files found in '{self.images_directory}' directory")
            return {}

        print(f"Found {len(jpg_files)} image(s) to process...")

        for image_path in jpg_files:
            # Extract name from filename (without extension)
            filename = os.path.basename(image_path)
            name = os.path.splitext(filename)[0]

            try:
                # Load and encode the face
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if len(face_encodings) > 0:
                    self.encodings[name] = face_encodings[0]
                    self.names.append(name)
                    print(f"✓ Loaded face encoding for '{name}' from {filename}")
                else:
                    print(f"✗ No face found in {filename}")

            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")

        print(f"Successfully loaded {len(self.encodings)} face encoding(s)")
        return self.encodings

    def get_encoding(self, name):
        """Get face encoding by name."""
        return self.encodings.get(name)

    def get_all_encodings(self):
        """Get all loaded encodings."""
        return self.encodings

    def get_names(self):
        """Get list of all loaded names."""
        return self.names