"""
Fonctions utilitaires pour le Smart Crosswalk Monitor
Avec gestion de zone polygone (4 points)
"""

import cv2
import numpy as np
from config import *


def point_in_polygon(x, y, polygon):
    """
    Test point dans polygone (algorithme ray-casting).
    polygon = liste de (x, y) points dans l'ordre.
    """
    n = len(polygon)
    inside = False
    if n < 3:
        return inside
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def get_crosswalk_polygon(frame_width, frame_height):
    """
    Convertit les 4 points en pourcentages vers des coordonnées pixels.
    Retourne: liste [(x, y), ...] de 4 points.
    """
    polygon = []
    for (x_pct, y_pct) in CROSSWALK_POLYGON_PERCENT:
        x = int(frame_width * x_pct / 100)
        y = int(frame_height * y_pct / 100)
        polygon.append((x, y))
    return polygon


def draw_crosswalk_polygon(frame, polygon, violation=False):
    """
    Dessine le polygone du passage piéton. Vert normal, rouge violation.
    """
    overlay = frame.copy()
    color = COLOR_RED if violation else COLOR_GREEN
    
    # Convertir en numpy array pour OpenCV
    pts = np.array(polygon, np.int32)
    
    # Polygone semi-transparent
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, CROSSWALK_OVERLAY_ALPHA, frame, 
                    1 - CROSSWALK_OVERLAY_ALPHA, 0, frame)
    
    # Contour
    cv2.polylines(frame, [pts], True, color, 2)
    
    # Label
    label = "!! VIOLATION ZONE !!" if violation else "CROSSWALK ZONE"
    x1, y1 = polygon[0]
    cv2.putText(frame, label, (x1, y1 - 10), FONT, 
                FONT_SCALE, color, FONT_THICKNESS)
    
    return frame


def is_in_zone(bbox, polygon, threshold=0.3):
    """
    Vérifie si la bounding box est "dans" le polygone.
    Teste si le centre de la bbox est dans le polygone.
    """
    bx1, by1, bx2, by2 = bbox
    center_x = (bx1 + bx2) / 2
    center_y = (by1 + by2) / 2
    return point_in_polygon(center_x, center_y, polygon)


def draw_detection(frame, bbox, label, color, is_violation=False):
    x1, y1, x2, y2 = [int(c) for c in bbox]
    thickness = 3 if is_violation else 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (text_w, text_h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), FONT, 
                FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
    return frame


def draw_status_panel(frame, persons_in_zone, vehicles_in_zone, 
                       violation, violation_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "SMART CROSSWALK MONITOR", (10, 25), 
                FONT, 0.8, COLOR_WHITE, 2)
    cv2.putText(frame, f"Pedestrians in zone: {persons_in_zone}", 
                (10, 55), FONT, 0.6, COLOR_YELLOW, 1)
    cv2.putText(frame, f"Vehicles in zone: {vehicles_in_zone}", 
                (10, 80), FONT, 0.6, COLOR_ORANGE, 1)
    cv2.putText(frame, f"Total Violations: {violation_count}", 
                (w - 300, 55), FONT, 0.7, COLOR_RED, 2)
    
    if violation:
        cv2.putText(frame, "!! VIOLATION DETECTED !!", 
                    (w - 350, 85), FONT, 0.7, COLOR_RED, 2)
    else:
        cv2.putText(frame, "Status: Normal", 
                    (w - 250, 85), FONT, 0.6, COLOR_GREEN, 1)
    return frame


def draw_violation_alert(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), COLOR_RED, 8)
    alert_text = "WARNING: Vehicle did not yield to pedestrian!"
    (text_w, text_h), _ = cv2.getTextSize(alert_text, FONT, 1.0, 2)
    cv2.rectangle(frame, (0, h - 60), (w, h), COLOR_RED, -1)
    text_x = (w - text_w) // 2
    cv2.putText(frame, alert_text, (text_x, h - 20), 
                FONT, 1.0, COLOR_WHITE, 2)
    return frame
