import cv2
import os
import time
import face_recognition
from ultralytics import YOLOE

def run():
    yoloe_model = YOLOE("yoloe-11s-seg.pt")
    names = ["person", "car", "bus", "bicycle", "phone", "laptop", "cup"]
    yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))

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
    yoloe_results = []

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

        # Run YOLOE every 5 frames instead of every frame
        if frame_count % 5 == 0:
            yoloe_results = yoloe_model.predict(frame, conf=0.3, verbose=False)

        # Draw stored YOLOE detections
        for r in yoloe_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = yoloe_model.model.names[cls]
                conf = box.conf[0]

                color = (255, 0, 0) if label == "person" else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Face recognition every 10 frames instead of 3
        if ian_face_encoding is not None and frame_count % 10 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)  # Even smaller
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # Use HOG (faster)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_data = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 5  # Scale back up
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

        cv2.imshow("YOLOE + Face Recognition", display_frame)

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