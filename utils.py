import cv2
import numpy as np
import config

def get_box_center(bbox):
    """Retourne le centre d'une bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(p1, p2):
    """Distance euclidienne entre deux points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

class VehicleTracker:
    """Tracker simple pour éviter de compter plusieurs fois la même violation."""
    def __init__(self, max_distance=100, max_frames_missing=30):
        self.vehicles = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
    
    def update(self, detections, polygon):
        new_violations = 0
        vehicles_in_zone = []
        matched_ids = set()
        
        for bbox in detections:
            center = get_box_center(bbox)
            in_zone = is_in_zone(bbox, polygon)
            
            best_id, best_dist = None, float('inf')
            
            for vid, vdata in self.vehicles.items():
                if vid in matched_ids: continue
                d = distance(center, vdata["center"])
                if d < best_dist and d < self.max_distance:
                    best_dist = d
                    best_id = vid
            
            if best_id is not None:
                matched_ids.add(best_id)
                self.vehicles[best_id]["center"] = center
                self.vehicles[best_id]["missing"] = 0
                self.vehicles[best_id]["bbox"] = bbox
                
                if in_zone and not self.vehicles[best_id]["violated"]:
                    self.vehicles[best_id]["violated"] = True
                    self.vehicles[best_id]["in_zone"] = True
                    new_violations += 1
                elif in_zone:
                    self.vehicles[best_id]["in_zone"] = True
                else:
                    self.vehicles[best_id]["in_zone"] = False
            else:
                self.vehicles[self.next_id] = {
                    "center": center, "bbox": bbox, 
                    "violated": False, "in_zone": in_zone, "missing": 0
                }
                if in_zone:
                    self.vehicles[self.next_id]["violated"] = True
                    new_violations += 1
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        ids_to_remove = []
        for vid in self.vehicles:
            if vid not in matched_ids:
                self.vehicles[vid]["missing"] += 1
                self.vehicles[vid]["in_zone"] = False
                if self.vehicles[vid]["missing"] > self.max_frames_missing:
                    ids_to_remove.append(vid)
                    
        for vid in ids_to_remove:
            del self.vehicles[vid]
        
        for vid, vdata in self.vehicles.items():
            if vdata.get("in_zone", False):
                vehicles_in_zone.append(vdata["bbox"])
                
        return new_violations, vehicles_in_zone

def point_in_polygon(x, y, polygon):
    """Algorithme ray-casting pour polygone à N côtés"""
    n = len(polygon)
    inside = False
    if n < 3: return inside
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

def get_crosswalk_polygon(frame_width, frame_height, polygon_percent=None):
    if polygon_percent is None:
        polygon_percent = config.CROSSWALK_POLYGON_PERCENT
    polygon = []
    for (x_pct, y_pct) in polygon_percent:
        x = int(frame_width * x_pct / 100)
        y = int(frame_height * y_pct / 100)
        polygon.append((x, y))
    return polygon

def is_in_zone(bbox, polygon):
    center = get_box_center(bbox)
    return point_in_polygon(center[0], center[1], polygon)

def draw_crosswalk_polygon(frame, polygon, violation=False):
    overlay = frame.copy()
    color = config.COLOR_RED if violation else config.COLOR_GREEN
    pts = np.array(polygon, np.int32)
    
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, config.CROSSWALK_OVERLAY_ALPHA, frame, 
                    1 - config.CROSSWALK_OVERLAY_ALPHA, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)
    
    label = "!! VIOLATION ZONE !!" if violation else "CROSSWALK ZONE"
    x1, y1 = polygon[0]
    cv2.putText(frame, label, (x1, y1 - 10), config.FONT, 
                config.FONT_SCALE, color, config.FONT_THICKNESS)
    return frame

def draw_detection(frame, bbox, label, color, is_violation=False):
    x1, y1, x2, y2 = [int(c) for c in bbox]
    thickness = 3 if is_violation else 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (text_w, text_h), _ = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), config.FONT, 
                config.FONT_SCALE, config.COLOR_WHITE, config.FONT_THICKNESS)
    return frame

def draw_status_panel(frame, persons_in_zone, vehicles_in_zone, violation, violation_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "SMART CROSSWALK MONITOR", (10, 25), 
                config.FONT, 0.8, config.COLOR_WHITE, 2)
    cv2.putText(frame, f"Pedestrians in zone: {persons_in_zone}", 
                (10, 55), config.FONT, 0.6, config.COLOR_YELLOW, 1)
    cv2.putText(frame, f"Vehicles in zone: {vehicles_in_zone}", 
                (10, 80), config.FONT, 0.6, config.COLOR_ORANGE, 1)
    cv2.putText(frame, f"Total Violations: {violation_count}", 
                (w - 300, 55), config.FONT, 0.7, config.COLOR_RED, 2)
    
    if violation:
        cv2.putText(frame, "!! VIOLATION DETECTED !!", 
                    (w - 350, 85), config.FONT, 0.7, config.COLOR_RED, 2)
    else:
        cv2.putText(frame, "Status: Normal", 
                    (w - 250, 85), config.FONT, 0.6, config.COLOR_GREEN, 1)
    return frame

def draw_violation_alert(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), config.COLOR_RED, 8)
    alert_text = "WARNING: Vehicle did not yield to pedestrian!"
    (text_w, text_h), _ = cv2.getTextSize(alert_text, config.FONT, 1.0, 2)
    cv2.rectangle(frame, (0, h - 60), (w, h), config.COLOR_RED, -1)
    text_x = (w - text_w) // 2
    cv2.putText(frame, alert_text, (text_x, h - 20), 
                config.FONT, 1.0, config.COLOR_WHITE, 2)
    return frame
