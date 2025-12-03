import os
import cv2
import mediapipe as mp
import csv
# near top of scripts/extract_features_advanced.py
ROOT = "datasets/behavior"   # <- IMPORTANT: path to cleaned behavior dataset
OUT_FRAMES = "features/frames"
OUT_SEQS = "features/sequences"
WINDOW = 30
STRIDE = 8

pose = mp.solutions.pose.Pose(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(frame):
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return [0] * (33 * 4)

    keypoints = []
    for lm in result.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return keypoints

def process_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_keypoints(frame))

    cap.release()
    return frames

def save_csv(frames, out_path, label):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in frames:
            writer.writerow(row + [label])

DATASET = "datasets/hockey_cleaned"
OUT = "features/hockey"

os.makedirs(OUT, exist_ok=True)

for class_name, label in [("fight", 1), ("nofight", 0)]:
    class_dir = os.path.join(DATASET, class_name)
    out_dir = os.path.join(OUT, class_name)
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(class_dir):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            input_path = os.path.join(class_dir, file)
            print("Processing:", input_path)

            frames = process_video(input_path)
            csv_path = os.path.join(out_dir, file.replace(".mp4", ".csv").replace(".avi", ".csv"))
            save_csv(frames, csv_path, label)

print("\n Hockey keypoint extraction completed!")
