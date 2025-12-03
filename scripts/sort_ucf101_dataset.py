import os
import shutil

BASE = "datasets/ucf101"

# ====== SELECTED CLASSES ====== #
FIGHT_CLASSES = ["BoxingPunchingBag", "BoxingSpeedBag", "Punch"]
WALK_CLASSES = ["WalkingWithDog"]
STAND_CLASSES = ["Typing", "WritingOnBoard", "BrushingTeeth"]

OUTPUT = {
    "fighting_punching": FIGHT_CLASSES,
    "walking_normal": WALK_CLASSES,
    "standing_normal": STAND_CLASSES,
}

SPLITS = ["train", "val", "test"]

def ensure_dirs():
    for out in OUTPUT.keys():
        for split in SPLITS:
            os.makedirs(f"datasets/cleaned_ucf101/{split}/{out}", exist_ok=True)

def copy_selected():
    for split in SPLITS:
        src_split_dir = os.path.join(BASE, split)
        print(f"\n=== PROCESSING {split.upper()} ===")

        for folder in os.listdir(src_split_dir):
            full_path = os.path.join(src_split_dir, folder)
            if not os.path.isdir(full_path):
                continue

            for label, class_list in OUTPUT.items():
                if folder in class_list:
                    dst_dir = f"datasets/cleaned_ucf101/{split}/{label}"
                    print(f"Copying {folder} → {dst_dir}")
                    shutil.copytree(full_path, os.path.join(dst_dir, folder), dirs_exist_ok=True)

ensure_dirs()
copy_selected()

print("\n DATASET SORTING COMPLETE!")
