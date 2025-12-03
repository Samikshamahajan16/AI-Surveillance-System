# ui/surveillance_core.py

import cv2
import numpy as np
import mediapipe as mp
import os
import datetime
import threading
import joblib
from catboost import CatBoostClassifier


class DetectionCore:
    def __init__(self, model_path, scaler_path, alert_folder, log_file):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.alert_folder = alert_folder
        self.log_file = log_file

        # -----------------------------
        # FIX: Load CatBoost model correctly
        # -----------------------------
        print("Loading CatBoost model (joblib):", model_path)
        self.model = joblib.load(model_path)

        # Load scaler
        print("Loading Scaler:", scaler_path)
        self.scaler = joblib.load(scaler_path)

        # Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # State
        self.running = False
        self.frame_bgr = None
        self.previous_kp = None

        # UI callback
        self.on_alert = None

        # NEW: current prediction for UI overlay
        self.current_label = "No Data"

        # Ensure folders
        os.makedirs(alert_folder, exist_ok=True)
        os.makedirs(os.path.join(alert_folder, "snapshots"), exist_ok=True)

    # -------------------------------------------------------
    # Start / Stop
    # -------------------------------------------------------
    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._run_loop, daemon=True).start()

    def stop(self):
        self.running = False

    # -------------------------------------------------------
    # Webcam Loop
    # -------------------------------------------------------
    def _run_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" ERROR: Could not open camera.")
            return

        LABELS = ["No Fight", "Fight"]
        alert_cooldown = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_bgr = frame.copy()

            # Extract pose keypoints
            kp = self.extract_pose(frame)
            features = self.build_features(kp)

            scaled_feat = self.scaler.transform(features.reshape(1, -1))

            # Predict using CatBoost
            pred = int(self.model.predict(scaled_feat)[0])
            label = LABELS[pred]

            #  New: update current label for UI
            self.current_label = label

            # Trigger alert
            if label == "Fight":
                if alert_cooldown == 0:
                    self.trigger_alert(frame, label)
                    alert_cooldown = 50

            if alert_cooldown > 0:
                alert_cooldown -= 1

        cap.release()

    # -------------------------------------------------------
    # Return last frame
    # -------------------------------------------------------
    def get_frame(self):
        return self.frame_bgr

    # -------------------------------------------------------
    # Pose Extraction
    # -------------------------------------------------------
    def extract_pose(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return np.zeros(132, dtype=np.float32)

        kp = []
        for lm in res.pose_landmarks.landmark:
            kp += [lm.x, lm.y, lm.z, lm.visibility]

        return np.array(kp, dtype=np.float32)

    # -------------------------------------------------------
    # Build 330-D Features
    # -------------------------------------------------------
    def build_features(self, curr):
        prev = self.previous_kp
        self.previous_kp = curr.copy()

        mean_f = curr
        std_f = np.zeros_like(curr)
        motion_f = np.zeros_like(curr) if prev is None else np.abs(curr - prev)

        return np.concatenate([mean_f, std_f, motion_f])

    # -------------------------------------------------------
    # Alerts
    # -------------------------------------------------------
    def trigger_alert(self, frame, label):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save snapshot
        snap_dir = os.path.join(self.alert_folder, "snapshots")
        snap_path = os.path.join(snap_dir, f"{label}_{timestamp}.jpg")
        cv2.imwrite(snap_path, frame)

        # Log event
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} — {label}\n")

        # UI callback
        if self.on_alert is not None:
            self.on_alert(frame, label, timestamp)
