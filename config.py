"""
Configuration du Smart Crosswalk Monitor
"""

# ============================================
# CLASSES YOLO (COCO dataset)
# ============================================
PERSON_CLASS_ID = 0       # 'person' dans COCO
CAR_CLASS_ID = 2          # 'car' dans COCO
TRUCK_CLASS_ID = 7        # 'truck' (optionnel)
BUS_CLASS_ID = 5          # 'bus' (optionnel)

# Toutes les classes véhicules à détecter
VEHICLE_CLASSES = [CAR_CLASS_ID, TRUCK_CLASS_ID, BUS_CLASS_ID]

# ============================================
# MODÈLE YOLO
# ============================================
YOLO_MODEL = "yolov8n.pt"  # nano = rapide, suffisant
CONFIDENCE_THRESHOLD = 0.5

# ============================================
# ZONE DU PASSAGE PIÉTON - 4 POINTS EN POURCENTAGES
# Format: [(x1%, y1%), (x2%, y2%), (x3%, y3%), (x4%, y4%)]
# Les points doivent être dans l'ordre (horaire ou anti-horaire)
# ============================================
CROSSWALK_POLYGON_PERCENT = [
    (93, 37),   # Point 1
    (100, 50),   # Point 2
    (9, 76),   # Point 3
    (1, 62)   # Point 4
]

# ============================================
# COULEURS (BGR pour OpenCV)
# ============================================
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 165, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 140, 255)

# ============================================
# AFFICHAGE
# ============================================
CROSSWALK_OVERLAY_ALPHA = 0.3  # Transparence de la zone
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
