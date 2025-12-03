import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from joblib import load
import time
import os

# Load XGBoost model + scaler
model = load("models/behavior_model_xgb.joblib")
scaler = load("models/behavior_scaler.joblib")

# Load YOLO for person detection
yolo = YOLO("yolov8n.pt")

# Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Create logs folder
os.makedirs("logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

def extract_pose(frame):
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return np.zeros(33*4, dtype=np.float32)

    pts = []
    for lm in result.pose_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(pts, dtype=np.float32)

def compute_angles(keypoints):
    try:
        kp = keypoints.reshape(33, 4)
        angles = []
        pairs = [
            (11, 13, 15),
            (12, 14, 16),
            (23, 11, 13),
            (24, 12, 14),
            (23, 25, 27),
            (24, 26, 28),
        ]
        for a, b, c in pairs:
            v1 = kp[a][:3] - kp[b][:3]
            v2 = kp[c][:3] - kp[b][:3]
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
            angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

        while len(angles) < 26:
            angles.append(0)

        return np.array(angles[:26], dtype=np.float32)

    except:
        return np.zeros(26, dtype=np.float32)

def compute_velocity(prev, curr):
    if prev is None:
        return np.zeros(33, dtype=np.float32)

    prev = prev.reshape(33, 4)
    curr = curr.reshape(33, 4)
    vel = np.linalg.norm(curr[:, :2] - prev[:, :2], axis=1)
    return vel.astype(np.float32)


# Real-time video
cap = cv2.VideoCapture(0)
prev_pose = None
fight_alert_shown = False

print(" AI Behavior Detection Running... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = yolo.predict(frame, imgsz=480, conf=0.5, classes=[0], verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person = frame[y1:y2, x1:x2]

            if person.size == 0:
                continue

            pose_kp = extract_pose(person)
            angles = compute_angles(pose_kp)
            velocity = compute_velocity(prev_pose, pose_kp)

            prev_pose = pose_kp.copy()

            feat = np.concatenate([pose_kp, angles, velocity])
            feat_scaled = scaler.transform([feat])

            pred = model.predict(feat_scaled)[0]

            if pred == 1:
                color = (0, 0, 255)
                label = " FIGHT DETECTED!"

                if not fight_alert_shown:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"snapshots/fight_{timestamp}.jpg", frame)
                    with open("logs/events.txt", "a") as f:
                        f.write(f"{timestamp}: Fight detected\n")
                    fight_alert_shown = True
            else:
                color = (0, 255, 0)
                label = "Normal"
                fight_alert_shown = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("AI Behavior Detection (XGBoost)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
