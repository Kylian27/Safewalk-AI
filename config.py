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
