import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from joblib import load
import time

# Load ML model
model = load("models/fight_behavior_model.joblib")

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # people detection

# MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Extract pose keypoints
def extract_keypoints(frame):
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return [0] * (33 * 4)
    keypoints = []
    for lm in result.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return keypoints

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = yolo_model.predict(frame, imgsz=480, conf=0.5, classes=[0], verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                continue

            keypoints = extract_keypoints(person)
            features = np.array(keypoints).reshape(1, -1)

            pred = model.predict(features)[0]

            if pred == 1:
                color = (0, 0, 255)  # red
                label = "FIGHT DETECTED!"
            else:
                color = (0, 255, 0)  # green
                label = "Normal"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("AI Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
