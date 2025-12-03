# AI-Surveillance-System

#  AI Surveillance System (Real-Time Behavior Detection)

An AI-powered intelligent video surveillance system that detects unusual human behaviors such as **fighting, running, and abnormal movements** using pose estimation and machine learning (CatBoost classifier).  
The system generates **real-time alerts**, captures **snapshots**, and displays everything on a modern **Tkinter UI dashboard** with an activity log and timeline.

---

##  Features

###  Real-Time Behavior Detection
- Detects **Fight** vs **No Fight** using pose landmarks  
- Fast frame-by-frame analysis using **MediaPipe Pose**

###  Machine Learning Model
- CatBoost Multi-Class Classifier  
- Uses **330-dimensional pose-motion features**  
- High accuracy and fast inference

###  Alert System
- Instant alert when fight is detected  
- Red flashing warning banner  
- Saves snapshot with timestamp  
- Adds entry into **Activity Log**

###  Modern UI Dashboard
- Live camera feed  
- Behavior label overlay  
- Activity log panel  
- Snapshot timeline with click-to-view  
- Smooth & non-freezing (thread-safe)

---

##  Tech Stack

| Component | Technology |
|----------|------------|
| Pose Detection | MediaPipe Pose |
| ML Model | CatBoost Classifier |
| Feature Engineering | NumPy |
| Video Processing | OpenCV |
| UI | Tkinter |
| Image Handling | Pillow |
| Threading | Python threads |

---

## 🔧 How It Works (Simple Explanation)

1. Capture live video stream  
2. Extract **33 human pose landmarks** (x, y, z, visibility)  
3. Build a **330-dimension feature vector**  
4. Scale & send to **CatBoost model**  
5. Predict: *Fight* or *No Fight*  
6. If fight detected:  
   - Save snapshot  
   - Flash alert banner  
   - Add to Activity Log  
   - Add snapshot to Timeline  
7. UI updates continuously without freezing

---

---

## ▶️ Run the Project

### 1️⃣ Activate Environment
```bash
venv\Scripts\activate
```
2️⃣ Start Application
```
python ui/app_presentation.py
```
Author

Samiksha Mahajan
Software Developer
