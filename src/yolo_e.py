import cv2
from ultralytics import YOLOE

def run():
    # Load YOLOE model
    model = YOLOE("yoloe-11s-seg.pt")

    # Set what you want to detect - easily change these!
    names = ["person", "red ball", "coffee cup", "phone", "laptop", "car", "bicycle"]
    model.set_classes(names, model.get_text_pe(names))

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("YOLOE Detection - Press 'q' to quit")
    print(f"Looking for: {', '.join(names)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOE detection
        results = model(frame, conf=0.3, verbose=False)

        # Get annotated frame with bounding boxes and labels
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOE - Describe Anything", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()