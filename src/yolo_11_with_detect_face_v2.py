import cv2
import os
import time
import json  # Add this import
import face_recognition
from ultralytics import YOLO

# MediaPipe fallback for faster face detection
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    print("Using MediaPipe for face detection (faster)")
except ImportError:
    USE_MEDIAPIPE = False
    print("MediaPipe not installed - using face_recognition")

# Add these helper functions
def load_config(config_path="src/detection_config.json"):
    """Load detection configuration from JSON file"""
    import os
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, config_path)
    print(f"Looking for config at: {full_path}")
    print(f"File exists: {os.path.exists(full_path)}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Config loaded successfully!")
        return config['detection_config']
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Detecting all objects.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

def get_enabled_classes(config):
    """Get list of enabled class IDs and their info"""
    if not config:
        return [], {}

    enabled_classes = []
    class_info = {}

    for obj_name, obj_config in config['enabled_objects'].items():
        if obj_config['enabled']:
            class_id = obj_config['class_id']
            enabled_classes.append(class_id)
            class_info[class_id] = {
                'name': obj_name,
                'color': tuple(obj_config['color'])
            }

    return enabled_classes, class_info

def run():
    # Switch to YOLO11 (Pi-ready, same functionality)
    yolo_model = YOLO("models/yolo11s.pt")  # Detects everything by default

    # Load configuration
    config = load_config()
    enabled_classes, class_info = get_enabled_classes(config)

    if enabled_classes:
        print(f"Whitelisting objects: {[info['name'] for info in class_info.values()]}")
    else:
        print("No config found - detecting all objects")

    reference_image_path = "images/ian.jpg"
    ian_face_encoding = None

    if os.path.exists(reference_image_path):
        ian_image = face_recognition.load_image_file(reference_image_path)
        ian_face_encodings = face_recognition.face_encodings(ian_image)
        if len(ian_face_encodings) > 0:
            ian_face_encoding = ian_face_encodings[0]
            print("Face recognition enabled for Ian")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    os.makedirs("output", exist_ok=True)

    prev_time = 0
    fps_values = []
    frame_count = 0
    face_data = []
    yolo_results = []

    print("Press 'q' to quit, 's' to save screenshot")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        current_time = time.time()
        if prev_time > 0:
            fps = 1 / (current_time - prev_time)
            fps_values.append(fps)
            if len(fps_values) > 5:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)
        else:
            avg_fps = 0
        prev_time = current_time

        display_frame = frame.copy()

        # Run YOLO every 5 frames
        if frame_count % 5 == 0:
            if enabled_classes:
                # Only detect whitelisted classes
                yolo_results = yolo_model.predict(frame, conf=0.3, verbose=False, classes=enabled_classes)
            else:
                # Detect all classes (fallback)
                yolo_results = yolo_model.predict(frame, conf=0.3, verbose=False)

        # Draw YOLO detections
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0]

                # Use custom name and color if available
                if cls in class_info:
                    label = class_info[cls]['name']
                    color = class_info[cls]['color']
                else:
                    label = yolo_model.names[cls]
                    color = (255, 0, 0) if label == "person" else (0, 255, 0)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Face recognition every 10 frames
        if ian_face_encoding is not None and frame_count % 10 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

            if USE_MEDIAPIPE:
                # MediaPipe face detection (much faster)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                mp_results = mp_face_detection.process(rgb_small_frame)

                face_data = []

                if mp_results.detections:
                    for detection in mp_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = small_frame.shape
                        x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                        w, h = int(bboxC.width * iw), int(bboxC.height * ih)

                        # Scale back up (5x since we used 0.2 resize)
                        left, top = x * 5, y * 5
                        right, bottom = (x + w) * 5, (y + h) * 5

                        # Extract face for recognition
                        face_img = small_frame[y:y+h, x:x+w]
                        if face_img.size > 0:
                            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            face_encodings = face_recognition.face_encodings(face_rgb)

                            if face_encodings:
                                matches = face_recognition.compare_faces([ian_face_encoding], face_encodings[0], tolerance=0.7)
                                face_distances = face_recognition.face_distance([ian_face_encoding], face_encodings[0])

                                name = "Unknown"
                                confidence = 0

                                if matches[0]:
                                    name = "Ian"
                                    confidence = (1 - face_distances[0]) * 100

                                face_data.append({
                                    'box': (left, top, right, bottom),
                                    'name': name,
                                    'confidence': confidence
                                })
            else:
                # Original face_recognition method
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_data = []

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    top *= 5
                    right *= 5
                    bottom *= 5
                    left *= 5

                    matches = face_recognition.compare_faces([ian_face_encoding], face_encoding, tolerance=0.7)
                    face_distances = face_recognition.face_distance([ian_face_encoding], face_encoding)

                    name = "Unknown"
                    confidence = 0

                    if matches[0]:
                        name = "Ian"
                        confidence = (1 - face_distances[0]) * 100

                    face_data.append({
                        'box': (left, top, right, bottom),
                        'name': name,
                        'confidence': confidence
                    })

        # Draw face recognition results
        for face in face_data:
            left, top, right, bottom = face['box']
            name = face['name']
            confidence = face['confidence']

            color = (0, 255, 255) if name == "Ian" else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 3)

            if name == "Ian":
                cv2.putText(display_frame, f"{name}: {confidence:.1f}%", (left, top - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(display_frame, name, (left, top - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("YOLO11 + Face Recognition", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"output/combined_capture_{timestamp}.jpg", display_frame)
            print(f"Screenshot saved")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()