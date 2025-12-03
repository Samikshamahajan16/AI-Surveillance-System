import cv2

CAM_INDEX = 0  

cap = cv2.VideoCapture(CAM_INDEX)

print("Press Q to quit")

if not cap.isOpened():
    print(f"Could not open camera index {CAM_INDEX}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
