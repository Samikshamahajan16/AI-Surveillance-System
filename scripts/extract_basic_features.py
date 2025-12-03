# scripts/extract_basic_features.py
import os
import cv2
import numpy as np
import mediapipe as mp

DATA_DIR = "datasets/hockey_fight"
OUT_DIR = "features"
os.makedirs(OUT_DIR, exist_ok=True)

FOLDERS = {
    "fight": os.path.join(DATA_DIR, "fight"),
    "nofight": os.path.join(DATA_DIR, "nofight"),
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def extract_pose_mediapipe(frame):
    """132 features: 33*(x,y,z,visibility). Uses zeros if missing."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return np.zeros(132, dtype=np.float32)

    kps = []
    for lm in res.pose_landmarks.landmark:
        kps += [lm.x, lm.y, lm.z, lm.visibility]

    return np.array(kps, dtype=np.float32)

def process_folder(label, folder_path):
    print(f"\n=== Processing {label.upper()} ===")

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi", ".mov"))]
    if len(files) == 0:
        print("No video files found!")
        return

    all_features = []

    for file in files:
        video_path = os.path.join(folder_path, file)
        print("Processing:", file)

        cap = cv2.VideoCapture(video_path)
        per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            key = extract_pose_mediapipe(frame)
            per_frame.append(key)

        cap.release()

        if len(per_frame) < 3:
            print("Skipping video (too short).")
            continue

        per_frame = np.stack(per_frame)

        # --- 330 FEATURE VECTOR ---
        mean_feat = per_frame.mean(axis=0)
        std_feat = per_frame.std(axis=0)
        motion_feat = np.abs(np.diff(per_frame, axis=0)).mean(axis=0)

        video_feature = np.concatenate([mean_feat, std_feat, motion_feat])  # 330 dimensions

        all_features.append(video_feature)

    if len(all_features) == 0:
        print(" No features extracted.")
        return

    out_path = os.path.join(OUT_DIR, f"{label}_features.npy")
    np.save(out_path, np.vstack(all_features))
    print(f" Saved: {out_path} — shape={np.vstack(all_features).shape}")

if __name__ == "__main__":
    process_folder("fight", FOLDERS["fight"])
    process_folder("nofight", FOLDERS["nofight"])
    print("\n Feature extraction completed.")
