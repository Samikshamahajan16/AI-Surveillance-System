# scripts/extract_advanced_features.py
"""
Extract advanced features per 30-frame clip from videos.
Outputs:
  features/clip_features.npy  (N x D)
  features/clip_labels.npy    (N,)
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from math import atan2, degrees

DATA_DIR = "datasets/hockey_fight"   # expects fight/ and nofight/
OUT_DIR = "features"
os.makedirs(OUT_DIR, exist_ok=True)

CLIP_LEN = 30    # frames per clip
STEP = 15        # overlap

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# helpers
def get_kps(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    kps = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
    return kps  # list of 33 (x,y,z)

def vec(a, b):
    return (b[0]-a[0], b[1]-a[1])

def angle_between(a,b,c):
    # angle at b formed by points a-b-c
    v1 = vec(b,a)
    v2 = vec(b,c)
    ang = atan2(v2[1], v2[0]) - atan2(v1[1], v1[0])
    ang = degrees(ang)
    ang = (ang + 360) % 360
    if ang > 180:
        ang = 360 - ang
    return ang

def torso_orientation(kps):
    # use left/right shoulder to compute torso angle (in degrees)
    # fallback using hip points if shoulders missing
    try:
        ls = kps[11]  # left_shoulder
        rs = kps[12]
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        return degrees(atan2(dy, dx))
    except:
        return 0.0

# angle list: (a_idx, b_idx, c_idx) typical: shoulder-elbow-wrist etc.
ANGLES = [
    (12, 14, 16),  # right shoulder-elbow-wrist
    (11, 13, 15),  # left shoulder-elbow-wrist
    (24, 26, 28),  # right hip-knee-ankle
    (23, 25, 27),  # left hip-knee-ankle
    (12, 24, 26),  # right shoulder-hip-knee (torso-leg)
    (11, 23, 25),  # left shoulder-hip-knee
]

def frame_features(kps):
    """Return vector of:
       - flattened normalized keypoints (33*3) if available else zeros
       - N angles (as above)
       - torso orientation
    """
    if kps is None:
        kp_flat = np.zeros(33*3, dtype=np.float32)
        angs = [0.0]*len(ANGLES)
        tor = 0.0
    else:
        kp_flat = np.array(kps).reshape(-1)  # 99
        angs = []
        for (a,b,c) in ANGLES:
            try:
                angs.append(angle_between(kps[a], kps[b], kps[c]))
            except:
                angs.append(0.0)
        tor = torso_orientation(kps)

    return np.concatenate([kp_flat, np.array(angs, dtype=np.float32), np.array([tor], dtype=np.float32)])

# process folder -> produce clip features and labels
def process_folder(label_name, folder_path):
    clip_feats = []
    clip_labels = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
    for fname in files:
        fp = os.path.join(folder_path, fname)
        cap = cv2.VideoCapture(fp)
        frames_feats = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kps = get_kps(frame)
            feats = frame_features(kps)  # length = 99 + len(ANGLES) +1
            frames_feats.append(feats)
        cap.release()
        if len(frames_feats) < CLIP_LEN:
            continue
        arr = np.stack(frames_feats)  # F x Df

        # sliding window --> produce clip features
        for start in range(0, arr.shape[0]-CLIP_LEN+1, STEP):
            clip = arr[start:start+CLIP_LEN]  # CLIP_LEN x Df
            # aggregate features for clip
            mean = clip.mean(axis=0)
            std = clip.std(axis=0)
            motion = np.abs(np.diff(clip, axis=0)).mean(axis=0)
            # angle-specific stats: ANGLES are at the tail of each frame_features
            # but we already included angles in mean/std/motion so full concatenation is fine
            clip_feat = np.concatenate([mean, std, motion])  # fixed length
            clip_feats.append(clip_feat)
            clip_labels.append(1 if label_name=="fight" else 0)

    return np.array(clip_feats, dtype=np.float32), np.array(clip_labels, dtype=np.int32)

if __name__ == "__main__":
    all_feats = []
    all_labels = []
    for lab, sub in [("fight", os.path.join(DATA_DIR, "fight")), ("nofight", os.path.join(DATA_DIR, "nofight"))]:
        if not os.path.exists(sub):
            print("Missing folder:", sub)
            continue
        feats, labs = process_folder(lab, sub)
        print(f"{lab}: clips={len(feats)} shape_each={feats.shape[1] if len(feats)>0 else 0}")
        if len(feats)>0:
            all_feats.append(feats)
            all_labels.append(labs)

    if len(all_feats)==0:
        raise SystemExit("No features extracted. Check dataset paths and video formats.")
    X = np.vstack(all_feats)
    y = np.concatenate(all_labels)

    np.save(os.path.join(OUT_DIR, "clip_features.npy"), X)
    np.save(os.path.join(OUT_DIR, "clip_labels.npy"), y)
    print("Saved clip_features.npy and clip_labels.npy ->", X.shape, y.shape)
