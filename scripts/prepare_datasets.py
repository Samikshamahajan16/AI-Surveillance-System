# scripts/prepare_datasets.py
import os, shutil

ROOT = "datasets"
OUT = os.path.join(ROOT, "behavior")   # destination
os.makedirs(OUT, exist_ok=True)

# map source folders -> target behavior folder name
MAPPING = {
    # UCF classes -> behavior labels
    "WalkingWithDog": "walking",
    "Walking": "walking",
    "Punch": "fighting",
    "BoxingPunchingBag": "fighting",
    "BoxingSpeedBag": "fighting",
    "Typing": "standing",
    "WritingOnBoard": "standing",
    "BrushingTeeth": "standing",
    "JumpingJack": "running",   # approximation
    "Lunges": "running",
    "FloorGymnastics": "falling",
    # hockey
    # hockey_fight/fight -> fighting, nofight -> standing/normal
}

# add hockey
HK = os.path.join(ROOT, "hockey_fight")
if os.path.exists(HK):
    os.makedirs(os.path.join(OUT, "fighting"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "standing"), exist_ok=True)
    for f in os.listdir(os.path.join(HK,"fight")):
        shutil.copy(os.path.join(HK,"fight",f), os.path.join(OUT,"fighting",f))
    for f in os.listdir(os.path.join(HK,"nofight")):
        shutil.copy(os.path.join(HK,"nofight",f), os.path.join(OUT,"standing",f))

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
                    shutil.copy(os.path.join(src_dir,file), os.path.join(dst_dir,file))
print("Dataset prepared at:", OUT)
