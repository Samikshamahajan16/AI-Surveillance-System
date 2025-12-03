import cv2
from ultralytics import YOLO

#  Use whatever index works for you (0, 1, ...)
CAM_INDEX = 0  

def main():
    print(" YOLO Pose test starting (normal, no brightening)...")

    # Load YOLO pose model (first run may download yolov8n-pose.pt)
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f" Could not open camera index {CAM_INDEX}")
        return

    print(" Camera opened. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Failed to grab frame")
            break

        # Run YOLO pose directly on the frame
        results = model.predict(frame, imgsz=640, conf=0.4)

        # Get annotated frame (with skeleton/keypoints drawn)
        annotated = results[0].plot()

        cv2.imshow("YOLO Pose (normal) - Press Q", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(" Q pressed, exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Done.")

if __name__ == "__main__":
    main()
