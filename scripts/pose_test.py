print(" pose_test.py is starting...")   

import cv2
import mediapipe as mp

print(" Imported cv2 and mediapipe")


CAM_INDEX = 0  

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

def main():
    print(f" Trying to open camera index {CAM_INDEX} ...")
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f" Could not open camera index {CAM_INDEX}")
        input("Press Enter to exit...")
        return

    print(" Camera opened. Press Q in the window to quit.")

    cv2.namedWindow("MediaPipe Pose - Press Q", cv2.WINDOW_NORMAL)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            print(" Failed to grab frame")
            break

        if frame_count % 60 == 0:
            # Print every ~60 frames so we know it's running
            print(f" Processing frame {frame_count}...")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("MediaPipe Pose - Press Q", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(" Q pressed, exiting loop.")
            break

    print(" Releasing camera and closing windows.")
    cap.release()
    cv2.destroyAllWindows()
    print(" Done.")

if __name__ == "__main__":
    main()
