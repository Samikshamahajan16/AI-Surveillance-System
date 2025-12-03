import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import winsound
import datetime
import threading
import time

# ======================
#  Flashing Alert Setup
# ======================
flash_state = False
last_flash_time = 0

def draw_alert(frame, text):
    global flash_state, last_flash_time

    # Flash interval (0.3 seconds)
    if time.time() - last_flash_time > 0.3:
        flash_state = not flash_state
        last_flash_time = time.time()

    if flash_state:
        # Red flashing border
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 40)

        # Big red alert box
        cv2.rectangle(frame, (50,30), (frame.shape[1]-50,150), (0,0,255), -1)

        # White bold alert text
        cv2.putText(frame, text,
                    (70,110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2,
                    (255,255,255),
                    5)

# ======================
#  Alarm Sound
# ======================
def play_alarm():
    try:
        winsound.PlaySound("alerts/alarm.wav", winsound.SND_FILENAME)
    except:
        pass

# ======================
#  Save Snapshot + Log
# ======================
def trigger_alert(frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save snapshot
    snap_path = f"alerts/snapshots/fight_{timestamp}.jpg"
    cv2.imwrite(snap_path, frame)

    # Log event
    with open("alerts/events.log", "a") as f:
        f.write(f"{timestamp} — FIGHT DETECTED\n")

    # Alarm thread
    threading.Thread(target=play_alarm, daemon=True).start()

# ======================
#  Load Model & Scaler
# ======================
model = joblib.load("models/xgb_behavior_full.joblib")
scaler = joblib.load("models/scaler_xgb_full.joblib")

LABELS = ["No Fight", "Fight"]

# ======================
# 🏃 MediaPipe Pose Setup
# ======================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Folders
os.makedirs("alerts/snapshots", exist_ok=True)

# ======================
#  Feature Extraction
# ======================
def extract_pose_mediapipe(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return np.zeros(132, dtype=np.float32)

    kps = []
    for lm in res.pose_landmarks.landmark:
        kps += [lm.x, lm.y, lm.z, lm.visibility]

    return np.array(kps, dtype=np.float32)

def get_330_features(prev, curr):
    mean_feat = curr
    std_feat = np.zeros_like(curr)

    if prev is None:
        motion_feat = np.zeros_like(curr)
    else:
        motion_feat = np.abs(curr - prev)

    return np.concatenate([mean_feat, std_feat, motion_feat])

# ======================
#  Start Camera
# ======================
cap = cv2.VideoCapture(0)
previous = None
alert_cooldown = 0

print(" AI Behavior Detection With Alerts Running... Press Q to exit")

# ======================
# 🔄 REAL-TIME LOOP
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract features
    kp = extract_pose_mediapipe(frame)
    feat330 = get_330_features(previous, kp)
    previous = kp.copy()

    feat330 = feat330.reshape(1, -1)
    feat_scaled = scaler.transform(feat330)

    # Predict
    pred = int(model.predict(feat_scaled)[0])
    label = LABELS[pred]

    # ======================
    #  ALERT SYSTEM
    # ======================
    if label == "Fight":
        if alert_cooldown == 0:
            trigger_alert(frame)
            alert_cooldown = 50

        # Use new flashing alert here
        draw_alert(frame, "ALERT: FIGHT DETECTED!")

    # Display label
    cv2.putText(frame, f"Behavior: {label}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if alert_cooldown > 0:
        alert_cooldown -= 1

    cv2.imshow("AI Surveillance (Alerts Enabled)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
