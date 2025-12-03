import cv2

print(" Checking camera indexes 0–10...")

for idx in range(10):
    cap = cv2.VideoCapture(idx)
    if cap.read()[0]:
        print(f" Camera index {idx} WORKS")
    else:
        print(f" Camera index {idx} unusable")
    cap.release()
