import cv2
import numpy as np
import config
import os
import time
import threading
from collections import deque

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
    x1, y1, x2, y2 = [int(c) for c in bbox]
    # bbox 꼭짓점 4개 + 하단 중심점 중 하나라도 폴리곤 안이면 True
    check_points = [
        ((x1 + x2) // 2, y2),   # 하단 중심
        (x1, y2),                # 하단 왼쪽
        (x2, y2),                # 하단 오른쪽
        ((x1 + x2) // 2, (y1 + y2) // 2),  # 중심
    ]
    return any(point_in_polygon(px, py, polygon) for px, py in check_points)

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

def draw_status_panel(frame, persons_in_zone, vehicles_in_zone, violation, violation_count, ego_status=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "SMART CROSSWALK MONITOR", (10, 25), 
                config.FONT, 0.8, config.COLOR_WHITE, 2)
    
    if ego_status is not None:
        cv2.putText(frame, f"Ego-car Status: {ego_status}", 
                    (10, 55), config.FONT, 0.6, config.COLOR_WHITE, 1)
        if "Monitoring" in ego_status or "Stopped" in ego_status:
            cv2.putText(frame, f"Pedestrians: {persons_in_zone} | Vehicles: {vehicles_in_zone}", 
                        (10, 80), config.FONT, 0.6, config.COLOR_YELLOW, 1)
        else:
            cv2.putText(frame, "Monitoring Suspended (Car Moving)", 
                        (10, 80), config.FONT, 0.6, config.COLOR_BLUE, 1)
    else:
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

class CameraMotionDetector:
    def __init__(self, max_features=150, motion_threshold=None, stop_frames_required=None, move_frames_required=None):
        self.max_features = max_features
        self.motion_threshold = motion_threshold if motion_threshold is not None else config.MOTION_THRESHOLD
        self.stop_frames_required = stop_frames_required if stop_frames_required is not None else config.STOP_FRAMES_REQUIRED
        self.move_frames_required = move_frames_required if move_frames_required is not None else config.MOVE_FRAMES_REQUIRED
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_pts = None
        self.consecutive_stop_frames = 0
        self.consecutive_move_frames = 0
        self.is_stopped = False

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Track features in the upper 70% of the frame (avoiding dashboard/hood at the bottom)
        roi = gray[0:int(h*0.7), :]
        
        motion = 0.0
        tracked = False
        
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )
            
            if next_pts is not None and status is not None:
                good_prev = self.prev_pts[status == 1]
                good_next = next_pts[status == 1]
                
                if len(good_prev) > 5:
                    displacements = np.linalg.norm(good_next - good_prev, axis=1)
                    # Median displacement handles outliers well (e.g. other moving objects)
                    motion = float(np.median(displacements))
                    tracked = True
                    self.prev_pts = good_next.reshape(-1, 1, 2)
                else:
                    tracked = False
                    
        if not tracked:
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10)
            if pts is not None:
                self.prev_pts = pts
            else:
                self.prev_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)
            motion = 0.0
            
        self.prev_gray = gray.copy()
        
        if tracked:
            if motion < self.motion_threshold:
                self.consecutive_stop_frames += 1
                self.consecutive_move_frames = 0
                if self.consecutive_stop_frames >= self.stop_frames_required:
                    self.is_stopped = True
            else:
                self.consecutive_move_frames += 1
                self.consecutive_stop_frames = 0
                if self.consecutive_move_frames >= self.move_frames_required:
                    self.is_stopped = False
                    
        return self.is_stopped, motion

class InfractionRecorder:
    def __init__(self, output_dir=None, buffer_before_sec=None, duration_after_sec=None, fps=30.0):
        self.output_dir = output_dir if output_dir is not None else config.INFRACTIONS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        before_sec = buffer_before_sec if buffer_before_sec is not None else config.VIDEO_BUFFER_BEFORE_SEC
        after_sec = duration_after_sec if duration_after_sec is not None else config.VIDEO_DURATION_AFTER_SEC
        
        self.fps = fps
        self.buffer_size_before = int(before_sec * fps)
        self.frames_after = int(after_sec * fps)
        
        self.frame_ring_buffer = deque(maxlen=self.buffer_size_before)
        self.active_sessions = {}
        self.session_lock = threading.Lock()

    def add_frame(self, frame):
        frame_copy = frame.copy()
        self.frame_ring_buffer.append(frame_copy)
        
        finished_sessions = []
        with self.session_lock:
            for session_id, session in list(self.active_sessions.items()):
                session["frames"].append(frame_copy)
                if len(session["frames"]) >= session["total_frames_target"]:
                    finished_sessions.append((session_id, session["frames"]))
                    del self.active_sessions[session_id]
                    
        for session_id, frames in finished_sessions:
            threading.Thread(
                target=self._save_video,
                args=(session_id, frames),
                daemon=True
            ).start()

    def trigger_violation(self, frame, vehicle_id=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        v_str = f"veh_{vehicle_id}" if vehicle_id is not None else "unknown"
        session_id = f"violation_{v_str}_{timestamp}"
        
        # Avoid duplicate overlapping sessions for the same vehicle in a short time
        with self.session_lock:
            for active_id in self.active_sessions:
                if f"violation_{v_str}_" in active_id:
                    return None
                    
            # Save screenshot immediately
            screenshot_path = os.path.join(self.output_dir, f"{session_id}.jpg")
            cv2.imwrite(screenshot_path, frame)
            
            # Start video session
            initial_frames = list(self.frame_ring_buffer)
            total_target = len(initial_frames) + self.frames_after
            
            self.active_sessions[session_id] = {
                "frames": initial_frames,
                "total_frames_target": total_target
            }
        return session_id

    def _save_video(self, session_id, frames):
        if not frames:
            return
        h, w, c = frames[0].shape
        video_path = os.path.join(self.output_dir, f"{session_id}.mp4")
        
        # Write mp4 video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()

