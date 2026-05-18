# Safewalk-AI

SafeWalk AI is a computer vision-based system designed to improve road safety by automatically monitoring pedestrian crosswalks. Using deep learning, it detects pedestrians and vehicles in real-time, tracks their movements, and automatically identifies dangerous situations where a vehicle fails to yield to a pedestrian.

## ✨ Key Features
Real-Time Detection: Utilizes YOLOv8 to accurately detect pedestrians and various types of vehicles (cars, trucks, buses).

Smart Tracking System: Implements custom bounding-box tracking to prevent double-counting vehicles and ensure accurate violation statistics.

Automated Violation Alerts: Triggers visual alerts and increments a violation counter when a vehicle enters the crosswalk zone while a pedestrian is present.

User-Friendly GUI: A built-in Tkinter dashboard allowing users to easily load videos, calibrate zones, and monitor the live feed without touching the code.

#### Dual Calibration Modes:

📍 Manual Calibration: Simply click 4 points on the video frame to define the crosswalk polygon.

🤖 Auto Calibration: Uses a custom Deep Learning segmentation model (EfficientNet-B4 Trimap) to automatically detect and draw the crosswalk boundaries.

#### 🛠️ Tech Stack
Language: Python 3

Computer Vision: OpenCV (cv2)

Object Detection: Ultralytics YOLOv8

Semantic Segmentation (Auto-Calib): PyTorch, Segmentation Models PyTorch

Graphical Interface: Tkinter, Pillow (PIL)

####🚀 Installation & Setup
Clone the repository:

```
git clone https://github.com/Kylian27/Safewalk-AI
cd SafeWalk-AI
```

Create a virtual environment (Recommended):

```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

Install the required dependencies:

```
pip install -r requirements.txt
```
(Ensure your requirements.txt includes: ultralytics, opencv-python, numpy, segmentation-models-pytorch, torch, torchvision, shapely)

Model Weights:
Ensure the auto-calibration model weights (base_best.pt) are located in the auto_calibrate/models/ directory. The YOLOv8 weights (yolov8n.pt) will be downloaded automatically on the first run.

#### 💻 How to Use
Launch the application:

```
python main.py
```
Load a Video: Click "📁 Open video" and select your .mp4 file.

Calibrate the Zone:

Click "📍 Manual Calibration" and click 4 corners on the image to draw the crosswalk.

OR click "🤖 Auto Calibration" to let the AI find the crosswalk for you.

Start Monitoring: Click "🚀 Start Detection" to run the YOLO model and watch the live analysis.

Stop: Click "⏹ Stop" to pause the detection.

SafeWalk AI is designed as a Supervisor Dashboard. In a real-world smart city deployment (Edge Computing / Cloud Architecture), fixed surveillance cameras would stream video feeds to a central server running this software. The system processes the feeds, logs violations, and can assist municipal authorities in identifying dangerous intersections that require better infrastructure or traffic enforcement.

#### 👥 Authors

Samy NASSET, Kylian LABRADOR, Yann MALARET and 윤치호
