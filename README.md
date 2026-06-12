# Safewalk-AI

SafeWalk AI is a computer vision-based system designed to improve road safety by automatically monitoring pedestrian crosswalks. Using deep learning, it detects pedestrians and vehicles in real-time, tracks their movements, and automatically identifies dangerous situations where a vehicle fails to yield to a pedestrian.

## ✨ Key Features

* **Real-Time Detection:** Utilizes YOLOv8 to accurately detect pedestrians and various types of vehicles (cars, trucks, buses).
* **Live IP Camera Support:** Built-in MJPEG stream reader optimized for smartphone cameras (e.g., DroidCam), allowing seamless wireless live feeds.
* **Moving Camera Mode (Dashcam / Ego-car):** An advanced mode utilizing Optical Flow (Lucas-Kanade) to detect camera motion. The system automatically suspends monitoring while moving and auto-calibrates a new crosswalk zone when the vehicle stops.
* **Automated Infraction Recorder (DVR):** Acts as an intelligent dashcam/CCTV. When a violation is detected, it automatically saves a snapshot (`.jpg`) and a buffered video clip (`.mp4`) showing the seconds leading up to and immediately following the event.
* **Smart Tracking System:** Implements a custom bounding-box tracking algorithm using Euclidean distance to prevent double-counting vehicles and ensure accurate violation statistics.
* **Multi-Threaded Architecture:** Dedicated threads for UI, live preview, and heavy YOLO processing ensure a responsive interface and smooth video playback, even on lower-end hardware.
* **User-Friendly GUI:** A built-in Tkinter dashboard allowing users to easily load videos, connect to IP streams, calibrate zones, view recordings, and monitor the live feed without touching the code.

#### 📍 Dual Calibration Modes
* **Manual Calibration:** Simply click 4 points on the video frame to define the crosswalk polygon. The system calculates percentage-based coordinates to automatically adapt to resolution changes (e.g., rotating a phone).
* **Auto Calibration:** Uses a custom Deep Learning segmentation model (EfficientNet-B4 Trimap) to automatically detect and draw the crosswalk boundaries.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Computer Vision:** OpenCV (cv2) for image processing and Optical Flow tracking.
* **Object Detection:** Ultralytics YOLOv8
* **Semantic Segmentation (Auto-Calib):** PyTorch, Segmentation Models PyTorch
* **Graphical Interface:** Tkinter, Pillow (PIL)
* **Geometry:** Custom ray-casting algorithms for precise polygon boundary checks.

## 🚀 Installation & Setup

Clone the repository:
```bash
git clone https://github.com/Kylian27/Safewalk-AI
cd SafeWalk-AI
```

Create a virtual environment (Recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(Ensure your requirements.txt includes: ultralytics, opencv-python, numpy, segmentation-models-pytorch, torch, torchvision, shapely)*

**Model Weights:**
Ensure the auto-calibration model weights (`base_best.pt`) are located in the `auto_calibrate/models/` directory. The YOLOv8 weights (`yolov8n.pt`) will be downloaded automatically on the first run.

## 💻 How to Use

Launch the application:
```bash
python main.py
```

1.  **Load a Source:** * Click "📁 Open video / camera".
    * Choose **Yes** to connect to a live DroidCam stream (enter your phone's IP address).
    * Choose **No** to load a local `.mp4`, `.avi`, or `.mkv` file.
2.  **Select Mode:**
    * Toggle **Moving Camera Mode** if you are using the system from inside a moving vehicle. The AI will handle the rest.
    * If unchecked (fixed camera), proceed to calibration.
3.  **Calibrate the Zone (Fixed Camera Mode):**
    * Click "📍 Manual Calibration" and click 4 corners on the image to draw the crosswalk.
    * OR click "🤖 Auto Calibration" to let the AI find the crosswalk for you.
4.  **Start Monitoring:** Click "🚀 Start Detection" to run the YOLO model and watch the live analysis.
5.  **Review Infractions:** Click "📂 View Recordings" to open the folder where violation videos and snapshots are automatically saved.
6.  **Stop:** Click "⏹ Stop Detection" to pause the AI and return to the live preview.

## ⚙️ Configuration
You can fine-tune the system's behavior by editing `config.py`. Key parameters include:
* `MOTION_THRESHOLD`: Adjusts the sensitivity of the moving camera detection.
* `VIDEO_BUFFER_BEFORE_SEC` / `VIDEO_DURATION_AFTER_SEC`: Controls the length of the auto-recorded violation clips.
* `CONFIDENCE_THRESHOLD`: Adjusts the strictness of the YOLOv8 detections.

---

*SafeWalk AI is designed as a Supervisor Dashboard. In a real-world smart city deployment (Edge Computing / Cloud Architecture), fixed surveillance cameras would stream video feeds to a central server running this software. The system processes the feeds, logs violations, and can assist municipal authorities in identifying dangerous intersections that require better infrastructure or traffic enforcement.*

#### 👥 Authors
Samy NASSET, Kylian LABRADOR, Yann MALARET and 윤치호