from ultralytics import YOLO

def main():
    model = YOLO("models/yolov8n.pt")

    results = model.predict(
        source=0,
        show=True,
        conf=0.25,
        imgsz=640
    )

if __name__ == "__main__":
    main()