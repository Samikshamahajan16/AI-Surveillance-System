import os

BASE_DIR = "datasets/ucf101"

for split in ["train", "val", "test"]:
    path = os.path.join(BASE_DIR, split)
    print(f"\n=== FOLDERS IN {split.upper()} ===")
    if not os.path.exists(path):
        print("  (missing)")
        continue
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            print(" -", name)
