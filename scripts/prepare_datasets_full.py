# scripts/prepare_datasets_full.py
import os, shutil

ROOT = "datasets"
OUT = os.path.join(ROOT, "behavior")   # destination
os.makedirs(OUT, exist_ok=True)

# map source UCF classes -> target behavior name
MAPPING = {
    "WalkingWithDog": "walking",
    "Walking": "walking",
    "WalkingWithDog": "walking",
    "Punch": "fighting",
    "BoxingPunchingBag": "fighting",
    "BoxingSpeedBag": "fighting",
    "Typing": "standing",
    "WritingOnBoard": "standing",
    "BrushingTeeth": "standing",
    "JumpingJack": "running",   # approximate running
    "Lunges": "running",
    "FloorGymnastics": "falling",
    "HandstandWalking": "running",
    "HighJump": "running",
}

# copy hockey fight -> fighting, nofight -> standing (normal)
HK = os.path.join(ROOT, "hockey_fight")
if os.path.exists(HK):
    os.makedirs(os.path.join(OUT, "fighting"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "standing"), exist_ok=True)
    src_fight = os.path.join(HK, "fight")
    src_nofight = os.path.join(HK, "nofight")
    if os.path.exists(src_fight):
        for f in os.listdir(src_fight):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                shutil.copy(os.path.join(src_fight,f), os.path.join(OUT,"fighting",f))
    if os.path.exists(src_nofight):
        for f in os.listdir(src_nofight):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                shutil.copy(os.path.join(src_nofight,f), os.path.join(OUT,"standing",f))

# handle UCF splits
UCF_BASE = os.path.join(ROOT, "ucf101")
for split in ["train", "val", "test"]:
    src = os.path.join(UCF_BASE, split)
    if not os.path.exists(src): continue
    for folder in os.listdir(src):
        if folder in MAPPING:
            tgt = MAPPING[folder]
            dst_dir = os.path.join(OUT, tgt)
            os.makedirs(dst_dir, exist_ok=True)
            src_dir = os.path.join(src, folder)
            for file in os.listdir(src_dir):
                if file.lower().endswith((".mp4",".avi",".mov")):
                    # avoid duplicate filenames
                    try:
                        shutil.copy(os.path.join(src_dir,file), os.path.join(dst_dir,file))
                    except Exception:
                        # if collision, prefix with folder
                        shutil.copy(os.path.join(src_dir,file), os.path.join(dst_dir, folder + "_" + file))

print("Dataset prepared at:", OUT)
