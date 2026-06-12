import cv2
import numpy as np
import config
import os
import time
import threading
from collections import deque

def get_box_center(bbox):
    """Calculates the center point of a bounding box given in the format (x1, y1, x2, y2). This function takes the coordinates of the top-left and bottom-right corners of the bounding box, computes the average of the x-coordinates and the average of the y-coordinates to find the center point. The resulting center point is returned as a tuple (center_x, center_y)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

class VehicleTracker:
    """Simple tracker to avoid counting the same violation multiple times."""
    def __init__(self, max_distance=100, max_frames_missing=30):
        self.vehicles = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
    
    def update(self, detections, polygon):
        """Updates the tracker with new detections and checks for violations. This method takes a list of detected bounding boxes and the crosswalk polygon as input. It attempts to match each detection to existing tracked vehicles based on proximity. If a detection is close enough to an existing vehicle, it updates that vehicle's information. If a detection is in the crosswalk zone and has not been marked as a violator, it marks it as a violation. The method also handles adding new vehicles to the tracker and removing vehicles that have been missing for too many frames. Finally, it returns the number of new violations detected in this update and a list of vehicles currently in the zone."""
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
    """Determines if a point (x, y) is inside a polygon defined by a list of vertices. This function uses the ray-casting algorithm to count how many times a horizontal ray, extending from the point to the right, intersects with the edges of the polygon. If the number of intersections is odd, the point is inside the polygon; if even, it is outside. The function returns True if the point is inside the polygon and False otherwise."""
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
    """Converts percentage-based polygon coordinates to pixel coordinates for the current video frame. This function takes the width and height of the video frame, as well as an optional list of polygon points defined as percentages of the frame dimensions. If no polygon_percent is provided, it uses the CROSSWALK_POLYGON_PERCENT from the config. It iterates through each point defined in percentage terms, calculates the corresponding pixel coordinates based on the frame dimensions, and returns a list of points that define the crosswalk polygon in pixel coordinates."""
    if polygon_percent is None:
        polygon_percent = config.CROSSWALK_POLYGON_PERCENT
    polygon = []
    for (x_pct, y_pct) in polygon_percent:
        x = int(frame_width * x_pct / 100)
        y = int(frame_height * y_pct / 100)
        polygon.append((x, y))
    return polygon

def is_in_zone(bbox, polygon):
    """Determines if a bounding box is inside the crosswalk polygon. This function checks if any of the key points of the bounding box (the bottom center, bottom left, bottom right, and center) are located within the defined polygon. It uses the point_in_polygon function to check each of these points against the polygon vertices. If any of these points are found to be inside the polygon, the function returns True, indicating that the bounding box is considered to be in the crosswalk zone; otherwise, it returns False."""
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
    """Draws the crosswalk polygon on the video frame. This function takes the video frame, a list of polygon points, and a boolean indicating whether there is a violation. It creates an overlay to draw the filled polygon with a color that indicates the status (red for violation, green for normal). It also draws the polygon outline and adds a label to indicate whether it is a violation zone or a normal crosswalk zone. The modified frame with the drawn polygon is returned for display."""
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
    """Draws a detection bounding box with a label on the video frame. This function takes the video frame, the bounding box coordinates, a label for the detected object, a color for the box and text, and a boolean indicating whether this detection is a violation. It draws a rectangle around the detected object, adds a filled rectangle for the label background, and puts the label text on top. The thickness of the bounding box is increased if it is a violation to make it more noticeable. The modified frame with the drawn detection is returned for display."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    thickness = 3 if is_violation else 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (text_w, text_h), _ = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), config.FONT, 
                config.FONT_SCALE, config.COLOR_WHITE, config.FONT_THICKNESS)
    return frame

def draw_status_panel(frame, persons_in_zone, vehicles_in_zone, violation, violation_count, ego_status=None):
    """Draws the status panel on the video frame. This function creates a semi-transparent overlay at the top of the video frame to display important information such as the current status of the ego-car, the number of pedestrians and vehicles in the crosswalk zone, and the total count of violations detected. It also indicates whether a violation is currently detected with a prominent message. The modified frame with the status panel is returned for display."""
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
    """Draws a violation alert on the video frame. This function creates a red border around the entire frame and displays a prominent warning message at the bottom of the screen to alert viewers that a vehicle has failed to yield to a pedestrian. The modified frame with the violation alert is returned for display."""
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
    """Detects if the camera (ego-car) is stopped or moving based on optical flow. This class uses the Lucas-Kanade method to track feature points across video frames and calculates the median motion of these points. If the median motion is below a certain threshold for a specified number of consecutive frames, it considers the camera to be stopped. If the motion exceeds the threshold for a certain number of consecutive frames, it considers the camera to be moving. The update method processes each new frame and returns the current status (stopped or moving) along with the calculated motion value."""
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
        """Processes a new video frame to determine if the camera is stopped or moving. This method converts the current frame to grayscale, detects and tracks feature points using optical flow, and calculates the median motion of these points. Based on the motion value and the number of consecutive frames that meet the stop or move criteria, it updates the internal state of whether the camera is considered stopped or moving. It returns a tuple indicating the current status (True for stopped, False for moving) and the calculated motion value."""
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
    """Records video clips of detected violations. This class maintains a ring buffer of recent video frames and manages active recording sessions for each detected violation. When a violation is triggered, it saves a screenshot immediately and starts a new recording session that includes the buffered frames before the violation and continues to record for a specified duration after the violation. The recorded video clips are saved in the specified output directory with unique filenames based on the timestamp and vehicle ID (if available). The class uses threading to save videos without blocking the main processing loop."""
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
        """Adds a new video frame to the ring buffer and updates active recording sessions. This method should be called for each new frame processed in the main loop. It appends the current frame to the ring buffer, then iterates through any active recording sessions and adds the frame to their respective buffers. If any session has reached its target number of frames (including both buffered frames before the violation and frames recorded after), it finalizes that session by starting a new thread to save the video clip and removes it from the active sessions list."""
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
        """Triggers a new violation recording session. This method should be called when a new violation is detected. It generates a unique session ID based on the current timestamp and the vehicle ID (if available), checks for existing active sessions for the same vehicle to avoid duplicates, saves a screenshot of the current frame immediately, and starts a new recording session that includes the buffered frames before the violation and continues to record for a specified duration after. The session information is stored in the active_sessions dictionary, which will be processed in subsequent calls to add_frame to build the video clip."""
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
        """Saves a video clip for a completed violation session. This method is run in a separate thread to avoid blocking the main processing loop. It takes the session ID and the list of frames that make up the video clip, checks if there are frames to save, and writes them to an MP4 file in the output directory with a filename based on the session ID."""
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

