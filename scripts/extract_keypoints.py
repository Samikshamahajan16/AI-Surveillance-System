import os
import cv2
import mediapipe as mp
import numpy as np
import csv

pose = mp.solutions.pose.Pose(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

def extract_pose_keypoints(frame):
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not result.pose_landmarks:
        return [0] * (33 * 4)  # x,y,z,visibility → 33 points

    keypoints = []
    for lm in result.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return keypoints


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kps = extract_pose_keypoints(frame)
        all_keypoints.append(kps)

    cap.release()
    return all_keypoints


def save_to_csv(keypoints, csv_path, label):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in keypoints:
            writer.writerow(row + [label])


def process_folder(dataset_root, out_folder, label):
    os.makedirs(out_folder, exist_ok=True)

    for file in os.listdir(dataset_root):
        if file.endswith((".mp4", ".avi", ".mov")):
            path = os.path.join(dataset_root, file)
            print("Processing:", path)
            kps = process_video(path)

            csv_name = file.replace(".mp4", ".csv").replace(".avi", ".csv")
            csv_path = os.path.join(out_folder, csv_name)

            save_to_csv(kps, csv_path, label)




DATASET_ROOT = "datasets/cleaned_ucf101"

CLASS_MAP = {
    "fighting_punching": 1,
    "walking_normal": 0,
    "standing_normal": 0
}

OUT_FOLDER = "features/ucf101"

for class_name, label in CLASS_MAP.items():
    folder_path = os.path.join(DATASET_ROOT, "train", class_name)
    out_path = os.path.join(OUT_FOLDER, class_name)
    process_folder(folder_path, out_path, label)

print("\nKeypoint extraction completed!")
