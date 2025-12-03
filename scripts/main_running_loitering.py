import cv2
import math
import os
import time
from datetime import datetime
from ultralytics import YOLO

CAM_INDEX = 0

LOG_DIR = "logs"
SNAP_DIR = "snapshots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# Models
det_model = YOLO("yolov8n.pt")       # person detection
pose_model = YOLO("yolov8n-pose.pt") # pose (if you want to also show skeleton)


def log_alert(message: str):
    path = os.path.join(LOG_DIR, "alerts.txt")
    with open(path, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {message}\n")
    print(f" {message}")


def save_snapshot(frame, label="alert"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SNAP_DIR, f"{label}_{ts}.jpg")
    cv2.imwrite(filename, frame)
    print(f" Snapshot saved: {filename}")


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f" Could not open camera index {CAM_INDEX}")
        return

    print(" AI Behavior Detection (running + loitering). Press Q to quit.")

    # Simple track state: id -> dict
    tracks = {}
    next_id = 0

    # thresholds (tune as needed)
    RUN_SPEED_THRESH = 80.0      # pixels per second (running)
    LOITER_TIME = 20.0           # seconds staying in area
    LOW_SPEED_THRESH = 10.0      # px/s considered "almost still"
    MATCH_DIST = 50.0            # px allowed to consider same person
    MAX_TRACK_AGE = 3.0          # remove track if unseen for > 3s

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Failed to grab frame")
            break

        now = time.time()
        dt = now - last_time
        if dt == 0:
            dt = 1e-3
        last_time = now

        #  Run YOLO person detection
        det_results = det_model(frame, imgsz=640, conf=0.5)
        det = det_results[0]

        new_centers = []  # (cx, cy, w, h)
        for box in det.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # 0 = person
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            new_centers.append((cx, cy, w, h))

        #  Match detections to existing tracks (naive nearest neighbor)
        used_track_ids = set()
        for (cx, cy, w, h) in new_centers:
            best_id = None
            best_dist = 1e9

            for tid, tr in tracks.items():
                dist = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if dist < best_dist and dist < MATCH_DIST:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                # create new track
                tracks[next_id] = {
                    "cx": cx,
                    "cy": cy,
                    "last_cx": cx,
                    "last_cy": cy,
                    "first_seen": now,
                    "last_seen": now,
                    "speed": 0.0,
                    "tag": "normal",  # normal / running / loitering
                }
                used_track_ids.add(next_id)
                next_id += 1
            else:
                # update existing track
                tr = tracks[best_id]
                tr["last_cx"] = tr["cx"]
                tr["last_cy"] = tr["cy"]
                tr["cx"] = cx
                tr["cy"] = cy
                tr["last_seen"] = now

                # speed in px/s
                dist = math.hypot(tr["cx"] - tr["last_cx"], tr["cy"] - tr["last_cy"])
                tr["speed"] = dist / dt
                used_track_ids.add(best_id)

        #  Remove old tracks
        to_remove = []
        for tid, tr in tracks.items():
            if now - tr["last_seen"] > MAX_TRACK_AGE:
                to_remove.append(tid)
        for tid in to_remove:
            tracks.pop(tid, None)

        #  Classify behavior for each track
        for tid, tr in tracks.items():
            time_alive = now - tr["first_seen"]
            speed = tr["speed"]

            behavior = "normal"

            if speed > RUN_SPEED_THRESH:
                behavior = "running"
            elif time_alive > LOITER_TIME and speed < LOW_SPEED_THRESH:
                behavior = "loitering"

            # if behavior changed from normal, log & snapshot
            if behavior != "normal" and behavior != tr["tag"]:
                msg = f"Track {tid}: {behavior.upper()} detected"
                log_alert(msg)
                save_snapshot(frame, label=behavior)

            tr["tag"] = behavior

            # draw on frame
            color = (0, 255, 0)
            if behavior == "running":
                color = (0, 0, 255)
            elif behavior == "loitering":
                color = (0, 255, 255)

            cv2.circle(frame, (int(tr["cx"]), int(tr["cy"])), 10, color, 2)
            cv2.putText(
                frame,
                f"ID {tid} {behavior}",
                (int(tr["cx"]) - 40, int(tr["cy"]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Optional: also draw pose skeleton for visual effect
        pose_results = pose_model(frame, imgsz=640, conf=0.4)
        annotated = pose_results[0].plot()

        # overlay tracking markers on annotated frame
        for tid, tr in tracks.items():
            color = (0, 255, 0)
            if tr["tag"] == "running":
                color = (0, 0, 255)
            elif tr["tag"] == "loitering":
                color = (0, 255, 255)

            cv2.circle(annotated, (int(tr["cx"]), int(tr["cy"])), 10, color, 2)
            cv2.putText(
                annotated,
                f"ID {tid} {tr['tag']}",
                (int(tr["cx"]) - 40, int(tr["cy"]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("AI Behavior Detection - running + loitering", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Stopped.")


if __name__ == "__main__":
    main()
