import cv2
import math
import os
import time
from datetime import datetime
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #

CAM_INDEX = 0  

LOG_DIR = "logs"
SNAP_DIR = "snapshots"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# Load models
det_model = YOLO("yolov8n.pt")          
pose_model = YOLO("yolov8n-pose.pt")    




def log_alert(message: str):
    """Write alert to log file"""
    path = os.path.join(LOG_DIR, "alerts.txt")
    with open(path, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {message}\n")
    print(f"{message}")


def save_snapshot(frame, label="alert"):
    """Save image snapshot"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SNAP_DIR, f"{label}_{ts}.jpg")
    cv2.imwrite(filename, frame)
    print(f" Snapshot saved: {filename}")


# ---------------- MAIN FUNCTION ---------------- #

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f" Could not open camera index {CAM_INDEX}")
        return

    print(" AI Behavior Detection Started (Running, Loitering, Fighting). Press Q to quit.")

    # Track dictionary
    tracks = {}  # id → info
    next_id = 0

    # Tunable thresholds
    RUN_SPEED_THRESH = 80.0       # px/s for running
    LOITER_TIME = 15.0            # seconds in same place
    LOW_SPEED_THRESH = 10.0       # px/s = still
    MATCH_DIST = 60.0             # px match to assign ID
    MAX_TRACK_AGE = 2.5           # seconds before wiping track

    FIGHT_DISTANCE = 90.0         # closeness threshold for fighting
    FIGHT_SPEED = 60.0            # speed of at least one person
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to grab frame")
            break

        now = time.time()
        dt = now - last_time
        if dt == 0:
            dt = 1e-3
        last_time = now

        #  Person Detection
        det_results = det_model(frame, imgsz=640, conf=0.5)
        detected_people = []

        for box in det_results[0].boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detected_people.append((cx, cy))

        #  Assign to tracks
        used = set()
        for (cx, cy) in detected_people:
            best_id = None
            best_dist = 1e9

            for tid, tr in tracks.items():
                dist = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if dist < best_dist and dist < MATCH_DIST:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                # new person
                tracks[next_id] = {
                    "cx": cx, "cy": cy,
                    "last_cx": cx, "last_cy": cy,
                    "speed": 0.0,
                    "first_seen": now,
                    "last_seen": now,
                    "tag": "normal"
                }
                used.add(next_id)
                next_id += 1
            else:
                # update person
                tr = tracks[best_id]
                tr["last_cx"] = tr["cx"]
                tr["last_cy"] = tr["cy"]
                tr["cx"] = cx
                tr["cy"] = cy
                tr["last_seen"] = now

                dist = math.hypot(tr["cx"] - tr["last_cx"], tr["cy"] - tr["last_cy"])
                tr["speed"] = dist / dt

                used.add(best_id)

        # remove lost tracks
        to_remove = []
        for tid, tr in tracks.items():
            if now - tr["last_seen"] > MAX_TRACK_AGE:
                to_remove.append(tid)

        for tid in to_remove:
            tracks.pop(tid, None)

        #  Behavior Classification (running, loitering)
        for tid, tr in tracks.items():
            time_alive = now - tr["first_seen"]
            speed = tr["speed"]
            behavior = "normal"

            if speed > RUN_SPEED_THRESH:
                behavior = "running"

            elif time_alive > LOITER_TIME and speed < LOW_SPEED_THRESH:
                behavior = "loitering"

            tr["behavior"] = behavior

        #  Fighting Detection
        fighting_pairs = []

        ids = list(tracks.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                p1, p2 = tracks[id1], tracks[id2]

                # distance between 2 persons
                dist = math.hypot(p1["cx"] - p2["cx"], p1["cy"] - p2["cy"])

                if dist < FIGHT_DISTANCE and (p1["speed"] > FIGHT_SPEED or p2["speed"] > FIGHT_SPEED):
                    fighting_pairs.append((id1, id2))

                    # log and snapshot once
                    if p1["behavior"] != "fighting" or p2["behavior"] != "fighting":
                        log_alert(f"Possible FIGHT detected between ID {id1} and ID {id2}")
                        save_snapshot(frame, label="fight")

                    p1["behavior"] = "fighting"
                    p2["behavior"] = "fighting"

        #  Pose overlay for nice visuals
        pose_results = pose_model(frame, imgsz=640, conf=0.4)
        annotated = pose_results[0].plot()

        #  Draw info on screen
        for tid, tr in tracks.items():
            x, y = int(tr["cx"]), int(tr["cy"])

            color = (0, 255, 0)
            if tr["behavior"] == "running":
                color = (0, 0, 255)
            elif tr["behavior"] == "loitering":
                color = (0, 255, 255)
            elif tr["behavior"] == "fighting":
                color = (0, 0, 150)

            cv2.circle(annotated, (x, y), 10, color, 2)
            cv2.putText(
                annotated,
                f"ID {tid} {tr['behavior']}",
                (x - 40, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("AI Behavior Detection (Running, Loitering, Fighting)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Stopped AI Behavior Detection")


if __name__ == "__main__":
    main()
