# ============================================
# YOLO classes
# ============================================
PERSON_CLASS_ID = 0       # 'person'
CAR_CLASS_ID = 2          # 'car'
TRUCK_CLASS_ID = 7        # 'truck'
BUS_CLASS_ID = 5          # 'bus'

# Classes to detect
VEHICLE_CLASSES = [CAR_CLASS_ID, TRUCK_CLASS_ID, BUS_CLASS_ID]

# ============================================
# YOLO model
# ============================================
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

# ============================================
# Crosswalk zone
# ============================================
CROSSWALK_POLYGON_PERCENT = []

# ============================================
# Colors
# ============================================
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 165, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 140, 255)

# ============================================
# Display
# ============================================
CROSSWALK_OVERLAY_ALPHA = 0.3
FONT = 0
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# ============================================
# Motion detection & moving camera settings
# ============================================
# Lowering this threshold makes the system more sensitive to slow movements.
# If the car is moving very slowly or in a straight line, a lower value (e.g., 0.6 - 0.8) helps.
# Avoid setting it too close to 0 to prevent camera noise/vibration from being seen as motion when stopped.
MOTION_THRESHOLD = 1.5
STOP_FRAMES_REQUIRED = 25
MOVE_FRAMES_REQUIRED = 8

# ============================================
# Infraction recorder settings
# ============================================
INFRACTIONS_DIR = "infractions"
VIDEO_BUFFER_BEFORE_SEC = 2.0
VIDEO_DURATION_AFTER_SEC = 3.0

