import cv2
import face_recognition

try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
except ImportError:
    USE_MEDIAPIPE = False

# Global variable to persist face_data between calls
_face_data = []

def recognize_faces(frame, known_encodings, frame_count):
    """
    Performs face recognition on the given frame.

    Args:
        frame (numpy.ndarray): The input frame.
        known_encodings (dict): Dictionary mapping names to face encodings.
        frame_count (int): The current frame count.

    Returns:
        numpy.ndarray: The modified display frame.
    """
    global _face_data
    display_frame = frame.copy()

    if known_encodings and frame_count % 10 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

        # Get lists of known encodings and names
        known_face_encodings = list(known_encodings.values())
        known_face_names = list(known_encodings.keys())

        if USE_MEDIAPIPE:
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_results = mp_face_detection.process(rgb_small_frame)

            _face_data = []

            if mp_results.detections:
                for detection in mp_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = small_frame.shape
                    x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                    w, h = int(bboxC.width * iw), int(bboxC.height * ih)

                    left, top = x * 5, y * 5
                    right, bottom = (x + w) * 5, (y + h) * 5

                    face_img = small_frame[y:y+h, x:x+w]
                    if face_img.size > 0:
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(face_rgb)

                        if face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.7)
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])

                            name = "Unknown"
                            confidence = 0

                            # Find the best match
                            if True in matches:
                                best_match_index = face_distances.argmin()
                                if matches[best_match_index]:
                                    name = known_face_names[best_match_index]
                                    confidence = (1 - face_distances[best_match_index]) * 100

                            _face_data.append({
                                'box': (left, top, right, bottom),
                                'name': name,
                                'confidence': confidence
                            })
        else:
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            _face_data = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 5
                right *= 5
                bottom *= 5
                left *= 5

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.7)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                name = "Unknown"
                confidence = 0

                # Find the best match
                if True in matches:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100

                _face_data.append({
                    'box': (left, top, right, bottom),
                    'name': name,
                    'confidence': confidence
                })

    # Draw the face data (this happens every frame, not just detection frames)
    for face in _face_data:
        left, top, right, bottom = face['box']
        name = face['name']
        confidence = face['confidence']

        # Use different colors for known vs unknown faces
        color = (0, 255, 255) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 3)

        if name != "Unknown":
            cv2.putText(display_frame, f"{name}: {confidence:.1f}%", (left, top - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(display_frame, name, (left, top - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return display_frame