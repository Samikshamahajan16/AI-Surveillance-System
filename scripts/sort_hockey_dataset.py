import os
import shutil

RAW_DIR = "datasets/hockey_fight"
FIGHT_DIR = os.path.join(RAW_DIR, "fight")
NOFIGHT_DIR = os.path.join(RAW_DIR, "nofight")

OUT_BASE = "datasets/hockey_cleaned"
OUT_FIGHT = os.path.join(OUT_BASE, "fight")
OUT_NOFIGHT = os.path.join(OUT_BASE, "nofight")

os.makedirs(OUT_FIGHT, exist_ok=True)
os.makedirs(OUT_NOFIGHT, exist_ok=True)

def copy_videos(src, dst):
    for file in os.listdir(src):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            shutil.copy(os.path.join(src, file), os.path.join(dst, file))
            print("Copied:", file)

copy_videos(FIGHT_DIR, OUT_FIGHT)
copy_videos(NOFIGHT_DIR, OUT_NOFIGHT)

print("\n✔ Hockey dataset sorted!")
