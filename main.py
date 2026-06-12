import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk
import threading
import json
import os
import time
import sys
import urllib.request
from queue import Queue, Empty
from ultralytics import YOLO
import utils
import config


class MJPEGStreamReader:
    """Reads MJPEG stream from a URL in a background thread, decodes JPEG frames, and provides the latest frame on demand."""
    def __init__(self, url: str):
        self.url = url
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
        self._connected = False
        self._frame_count = 0

    def start(self) -> tuple[bool, object]:
        """Starts the reader thread and waits for the first frame to be available."""
        self._running = True
        self._connected = False
        self._latest_frame = None
        first_frame_event = threading.Event()
        first_frame_holder = [None]
        self._thread = threading.Thread(
            target=self._reader_loop,
            args=(first_frame_event, first_frame_holder),
            daemon=True,
            name="MJPEGReader"
        )
        self._thread.start()
        got_frame = first_frame_event.wait(timeout=10.0)
        if not got_frame or first_frame_holder[0] is None:
            self._running = False
            return False, None
        return True, first_frame_holder[0].copy()

    def get_frame(self) -> tuple[bool, object]:
        """Retrieves the latest frame from the stream."""
        with self._lock:
            if self._latest_frame is None or not self._connected:
                return False, None
            return True, self._latest_frame.copy()

    def stop(self):
        """Stops the reader thread."""
        self._running = False
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._lock:
            self._latest_frame = None

    @property
    def is_connected(self) -> bool:
        """Returns whether the reader is connected to the stream."""
        return self._connected

    @property
    def frame_count(self) -> int:
        """Returns the number of frames received."""
        return self._frame_count

    def _reader_loop(self, first_frame_event, first_frame_holder):
        """Continuously reads from the MJPEG stream, decodes JPEG frames, and updates the latest frame."""
        first_frame_sent = False
        while self._running:
            try:
                print(f"[MJPEG] Connecting to {self.url}")
                req = urllib.request.Request(
                    self.url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Connection": "keep-alive",
                        "Cache-Control": "no-cache",
                    }
                )
                response = urllib.request.urlopen(req, timeout=15)
                content_type = response.headers.get("Content-Type", "")
                print(f"[MJPEG] Connected. Content-Type: {content_type}")
                self._connected = True
                self._stream_jpeg_scan(
                    response, first_frame_event, first_frame_holder, first_frame_sent
                )
                first_frame_sent = (first_frame_holder[0] is not None)
            except Exception as e:
                print(f"[MJPEG] Error: {e}")
                self._connected = False
                if not first_frame_sent:
                    first_frame_event.set()
                    return
            if self._running:
                self._connected = False
                print("[MJPEG] Stream lost. Waiting 1s before reconnect...")
                for _ in range(10):
                    if not self._running:
                        return
                    time.sleep(0.1)
        print("[MJPEG] Reader stopped.")

    def _stream_jpeg_scan(self, response, first_frame_event,
                          first_frame_holder, first_frame_sent):
        """Scans the MJPEG stream for JPEG frames and decodes them."""
        JPEG_START = b"\xff\xd8"
        JPEG_END = b"\xff\xd9"
        CHUNK_SIZE = 8192
        buf = b""
        frames_decoded = 0
        while self._running:
            try:
                chunk = response.read(CHUNK_SIZE)
            except Exception as e:
                print(f"[MJPEG] Read error: {e}")
                break
            if not chunk:
                print("[MJPEG] Server closed the connection.")
                break
            buf += chunk
            while True:
                start = buf.find(JPEG_START)
                if start == -1:
                    buf = buf[-1:]
                    break
                end = buf.find(JPEG_END, start + 2)
                if end == -1:
                    buf = buf[start:]
                    break
                jpeg_bytes = buf[start:end + 2]
                buf = buf[end + 2:]
                frame = self._decode_jpeg(jpeg_bytes)
                if frame is None:
                    continue
                frames_decoded += 1
                self._frame_count += 1
                with self._lock:
                    self._latest_frame = frame
                if not first_frame_sent:
                    first_frame_holder[0] = frame.copy()
                    first_frame_event.set()
                    first_frame_sent = True
                    print(f"[MJPEG] First frame: {frame.shape}")
                if frames_decoded % 30 == 0:
                    print(f"[MJPEG] {self._frame_count} total frames received")

    def _decode_jpeg(self, data: bytes):
        """Decodes JPEG bytes into an OpenCV image."""
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None


class CaptureManager:
    """Manages video capture from either an MJPEG stream or a video file, providing a unified interface for frame retrieval."""
    def __init__(self):
        self.source = None
        self.is_live = False
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0
        self._mjpeg_reader = None
        self._file_cap = None

    def open(self, source: str, is_live: bool) -> tuple[bool, object]:
        """Opens the specified video source, which can be either a live MJPEG stream or a video file. Returns a tuple indicating success and the first frame (if successful)."""
        self.close()
        self.source = source
        self.is_live = is_live
        if is_live:
            return self._open_live(source)
        else:
            return self._open_file(source)

    def get_frame(self) -> tuple[bool, object]:
        """Retrieves the latest frame from the video source."""
        if self.is_live:
            if self._mjpeg_reader is None:
                return False, None
            return self._mjpeg_reader.get_frame()
        else:
            if self._file_cap is None or not self._file_cap.isOpened():
                return False, None
            return self._file_cap.read()

    def get_snapshot(self) -> tuple[bool, object]:
        """Gets a snapshot frame without advancing the video file position (for file sources) or just the latest frame for live sources."""
        if self.is_live:
            if self._mjpeg_reader is None:
                return False, None
            return self._mjpeg_reader.get_frame()
        else:
            if self._file_cap is None or not self._file_cap.isOpened():
                return False, None
            pos = self._file_cap.get(cv2.CAP_PROP_POS_FRAMES)
            self._file_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._file_cap.read()
            self._file_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            return ret, frame

    def get_mjpeg_frame_count(self) -> int:
        """Returns the number of frames received from the MJPEG stream. For file sources, this returns 0."""
        if self._mjpeg_reader is not None:
            return self._mjpeg_reader.frame_count
        return 0

    def reset_to_start(self):
        """Resets the video file to the beginning. For live sources, this does nothing."""
        if not self.is_live and self._file_cap is not None:
            self._file_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        """Closes any open video source and releases resources."""
        if self._mjpeg_reader is not None:
            self._mjpeg_reader.stop()
            self._mjpeg_reader = None
        if self._file_cap is not None:
            self._file_cap.release()
            self._file_cap = None

    @property
    def is_open(self) -> bool:
        """Returns whether a video source is currently open and connected."""
        if self.is_live:
            return self._mjpeg_reader is not None and self._mjpeg_reader.is_connected
        else:
            return self._file_cap is not None and self._file_cap.isOpened()

    def _open_live(self, source: str) -> tuple[bool, object]:
        """Attempts to connect to the specified MJPEG stream URL and retrieves the first frame to determine resolution and confirm connectivity."""
        reader = MJPEGStreamReader(source)
        ret, frame = reader.start()
        if not ret or frame is None:
            return False, None
        self._mjpeg_reader = reader
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        self.fps = 30.0
        return True, frame

    def _open_file(self, source: str) -> tuple[bool, object]:
        """Attempts to open the specified video file and retrieves the first frame to determine resolution, FPS, and confirm that the file is valid."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return False, None
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, None
        self._file_cap = cap
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True, frame


class AutoCalibrator:
    """Handles automatic calibration of the crosswalk polygon using a pre-trained model. Loads the model on demand and provides thread-safe access to calibration functionality."""
    def __init__(self):
        self.model = None
        self.predict_trimap = None
        self.extract_polygon = None
        self.lock = threading.Lock()

    def load(self):
        """Loads the auto-calibration model and utilities if they haven't been loaded already. This is done in a thread-safe manner to ensure that the model is only loaded once, even if multiple threads attempt to calibrate at the same time."""
        with self.lock:
            if self.model is not None:
                return
            auto_calib_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), 'auto_calibrate')
            )
            root_config = sys.modules.get('config')
            root_utils = sys.modules.get('utils')
            if 'config' in sys.modules: del sys.modules['config']
            if 'utils' in sys.modules: del sys.modules['utils']
            sys.path.insert(0, auto_calib_dir)
            try:
                from poly_utils import load_model, predict_trimap, extract_polygon
                import config as ac_config
                self.model = load_model(ac_config.FINETUNE_CKPT)
                self.predict_trimap = predict_trimap
                self.extract_polygon = extract_polygon
            finally:
                sys.path.pop(0)
                if root_config: sys.modules['config'] = root_config
                if root_utils: sys.modules['utils'] = root_utils

    def calibrate(self, frame):
        """Performs auto-calibration on the given frame. Loads the model if it hasn't been loaded yet, then uses the model to predict a trimap and extract the crosswalk polygon from it. Returns the polygon as a list of percentage coordinates relative to the frame size, or None if calibration fails."""
        self.load()
        if self.model is None:
            return None
        trimap = self.predict_trimap(self.model, frame)
        return self.extract_polygon(trimap)


class LivePreviewThread(threading.Thread):
    """Continuously captures frames from the video source and updates the preview display. Draws a polygon overlay if one is set, and shows FPS and connection status. This thread runs independently of the main processing thread to ensure that the UI remains responsive even if processing is intensive."""
    def __init__(self, capture_manager, display_queue, stop_event,
                 canvas_width, canvas_height):
        super().__init__(name="LivePreview", daemon=True)
        self.capture_manager = capture_manager
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        # Shared polygon for overlay — set from main thread
        self.polygon_percent = []
        self._polygon_lock = threading.Lock()

    def set_polygon(self, polygon_percent: list):
        """Sets the polygon to be drawn as an overlay on the preview. The polygon is defined as a list of (x_percent, y_percent) tuples, where the coordinates are relative to the frame size. This method is thread-safe and can be called from the main thread while the preview thread is running."""
        with self._polygon_lock:
            self.polygon_percent = list(polygon_percent)

    def run(self):
        """Main loop for the live preview thread. Continuously captures frames from the video source, draws the polygon overlay if set, and updates the display queue with the latest frame. Also calculates and displays FPS and connection status. The loop runs until the stop event is set."""
        print("[Preview] Started")
        last_frame_count = -1
        fps_counter = FPSCounter()

        while not self.stop_event.is_set():
            cm = self.capture_manager

            if cm.is_live:
                current_count = cm.get_mjpeg_frame_count()
                if current_count == last_frame_count:
                    time.sleep(0.01)
                    continue
                last_frame_count = current_count

            ret, frame = cm.get_frame()
            if not ret or frame is None:
                if cm.is_live:
                    time.sleep(0.033)
                    continue
                else:
                    cm.reset_to_start()
                    time.sleep(0.033)
                    continue

            fps = fps_counter.tick()

            # Draw calibration polygon overlay if one exists
            with self._polygon_lock:
                poly = list(self.polygon_percent)

            if poly and len(poly) >= 3:
                polygon_px = utils.get_crosswalk_polygon(
                    cm.frame_width, cm.frame_height, poly
                )
                frame = utils.draw_crosswalk_polygon(frame, polygon_px, False)

            # Draw FPS and connection status
            frame = self._draw_overlay(frame, fps, cm.is_live)

            frame_resized = cv2.resize(
                frame, (self.canvas_width, self.canvas_height),
                interpolation=cv2.INTER_LINEAR
            )
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)

            if not self.display_queue.full():
                self.display_queue.put(img)

        print("[Preview] Stopped")

    def _draw_overlay(self, frame, fps, is_live):
        """Draws the FPS and connection status overlay on the given frame."""
        h, w = frame.shape[:2]
        source_text = "LIVE" if is_live else "FILE"
        color = (0, 200, 0) if is_live else (200, 200, 0)
        cv2.putText(
            frame, f"{source_text} | {fps:.1f} FPS",
            (w - 200, 30), config.FONT, 0.7, color, 2
        )
        cv2.putText(
            frame, "PREVIEW MODE — No detection running",
            (10, h - 15), config.FONT, 0.6, (200, 200, 200), 1
        )
        return frame


class FPSCounter:
    """Simple utility class to calculate frames per second (FPS) over a sliding window of time. Each call to tick() records the current time and returns the average FPS based on the number of ticks and the elapsed time between the first and last tick in the window."""
    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


class VideoProcessorThread(threading.Thread):
    """Main processing thread that handles running the YOLO model on video frames, tracking vehicles and pedestrians, detecting violations, and preparing annotated frames for display. This thread runs independently of the live preview thread to ensure that the UI remains responsive even if processing is intensive. It also manages the state for moving camera mode, including auto-calibration and infraction recording."""
    # Max frames per second to send to YOLO.
    # Frames arriving faster than this are skipped for YOLO
    # but the latest frame is always shown.
    YOLO_MAX_FPS = 12

    def __init__(self, capture_manager, model, polygon_percent, frame_queue,
                 stop_event, canvas_width, canvas_height,
                 moving_camera_mode=False, on_disconnect=None,
                 on_violation_saved=None):
        super().__init__(name="VideoProcessor", daemon=True)
        self.capture_manager = capture_manager
        self.model = model
        self.polygon_percent = polygon_percent
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.moving_camera_mode = moving_camera_mode
        self.on_disconnect = on_disconnect
        self.on_violation_saved = on_violation_saved
        self.tracker = utils.VehicleTracker(max_distance=100, max_frames_missing=30)
        self.violation_count = 0
        self._fps_counter = FPSCounter()
        self._yolo_fps_counter = FPSCounter()

    def run(self):
        """Main loop for the video processing thread. Continuously captures frames from the video source, runs YOLO detection at a throttled rate, updates vehicle tracking and violation detection, and prepares annotated frames for display. Also manages the state for moving camera mode, including auto-calibration and infraction recording. The loop runs until the stop event is set or the video stream ends."""
        cm = self.capture_manager
        print(f"[Processor] Started — is_live={cm.is_live} "
              f"size={cm.frame_width}x{cm.frame_height} fps={cm.fps}")

        frame_width = cm.frame_width
        frame_height = cm.frame_height
        fps = cm.fps

        cm.reset_to_start()

        if self.moving_camera_mode:
            self.infraction_recorder = utils.InfractionRecorder(fps=fps)
            self.motion_detector = utils.CameraMotionDetector()
            self.auto_calibrator = AutoCalibrator()
            ego_state = "Moving"
            polygon = []
            self.polygon_percent = []
            calib_retry_counter = 0
            # Run auto-calibration in a separate thread so it doesn't freeze the video
            self._calib_lock = threading.Lock()
            self._calib_running = False
            self._calib_result = None
        else:
            polygon = utils.get_crosswalk_polygon(
                frame_width, frame_height, self.polygon_percent
            )
            ego_state = None

        last_frame_count = -1
        frame_count = 0
        last_yolo_time = 0.0
        yolo_interval = 1.0 / self.YOLO_MAX_FPS

        # Track disconnection
        consecutive_no_frame = 0
        MAX_NO_FRAME = 60

        while not self.stop_event.is_set():
            if cm.is_live:
                current_count = cm.get_mjpeg_frame_count()
                if current_count == last_frame_count:
                    time.sleep(0.01)
                    continue
                last_frame_count = current_count

            ret, frame = cm.get_frame()

            if not ret or frame is None:
                if cm.is_live:
                    consecutive_no_frame += 1
                    if consecutive_no_frame >= MAX_NO_FRAME:
                        print("[Processor] Stream lost — notifying UI")
                        if self.on_disconnect:
                            self.on_disconnect()
                        break
                    time.sleep(0.033)
                    continue
                else:
                    print("[Processor] End of file")
                    break

            consecutive_no_frame = 0
            frame_count += 1
            display_fps = self._fps_counter.tick()

            # Handle resolution change mid-stream (phone rotation)
            h, w = frame.shape[:2]
            if w != frame_width or h != frame_height:
                print(f"[Processor] Resolution changed: {frame_width}x{frame_height}"
                      f" -> {w}x{h}")
                frame_width, frame_height = w, h
                if not self.moving_camera_mode and len(self.polygon_percent) >= 3:
                    polygon = utils.get_crosswalk_polygon(
                        frame_width, frame_height, self.polygon_percent
                    )

            if self.moving_camera_mode:
                ego_state, polygon, calib_retry_counter = self._handle_moving_camera(
                    frame, frame_width, frame_height,
                    ego_state, polygon, calib_retry_counter
                )

            # Throttle YOLO to YOLO_MAX_FPS
            now = time.time()
            run_yolo = (now - last_yolo_time) >= yolo_interval
            monitoring_active = (
                not self.moving_camera_mode or ego_state == "Stopped & Monitoring"
            )

            persons_in_zone = 0
            person_bboxes = []
            vehicle_bboxes = []
            vehicles_in_zone_bboxes = []

            if monitoring_active and run_yolo:
                last_yolo_time = now
                self._yolo_fps_counter.tick()
                results = self.model(
                    frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False
                )
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    if cls_id == config.PERSON_CLASS_ID:
                        if len(polygon) >= 3 and utils.is_in_zone(bbox, polygon):
                            persons_in_zone += 1
                            person_bboxes.append(
                                (bbox, f"Ped {conf:.0%} [IN]", config.COLOR_YELLOW)
                            )
                        else:
                            person_bboxes.append(
                                (bbox, f"Ped {conf:.0%}", config.COLOR_GREEN)
                            )
                    elif cls_id in config.VEHICLE_CLASSES:
                        class_name = self.model.names[cls_id]
                        vehicle_bboxes.append((bbox, class_name, conf))

                if len(polygon) >= 3:
                    vehicle_detections = [v[0] for v in vehicle_bboxes]
                    if persons_in_zone > 0:
                        new_violations, vehicles_in_zone_bboxes = self.tracker.update(
                            vehicle_detections, polygon
                        )
                        self.violation_count += new_violations
                    else:
                        _, vehicles_in_zone_bboxes = self.tracker.update(
                            vehicle_detections, polygon
                        )

            current_violation = (
                monitoring_active
                and persons_in_zone > 0
                and len(vehicles_in_zone_bboxes) > 0
            )

            processed_frame = frame.copy()

            for bbox, label, color in person_bboxes:
                processed_frame = utils.draw_detection(
                    processed_frame, bbox, label, color
                )

            for bbox, class_name, conf in vehicle_bboxes:
                in_zone = monitoring_active and (
                    bbox in vehicles_in_zone_bboxes
                    or (len(polygon) >= 3 and utils.is_in_zone(bbox, polygon))
                )
                if in_zone and persons_in_zone > 0:
                    label = f"{class_name} {conf:.0%} [VIOLATION]"
                    color = config.COLOR_RED
                    is_viol = True
                elif in_zone:
                    label = f"{class_name} {conf:.0%} [IN ZONE]"
                    color = config.COLOR_ORANGE
                    is_viol = False
                else:
                    label = f"{class_name} {conf:.0%}"
                    color = config.COLOR_BLUE
                    is_viol = False
                processed_frame = utils.draw_detection(
                    processed_frame, bbox, label, color, is_violation=is_viol
                )

            if len(polygon) >= 3:
                processed_frame = utils.draw_crosswalk_polygon(
                    processed_frame, polygon, current_violation
                )

            processed_frame = utils.draw_status_panel(
                processed_frame,
                persons_in_zone,
                len(vehicles_in_zone_bboxes),
                current_violation,
                self.violation_count,
                ego_status=ego_state
            )

            if current_violation:
                processed_frame = utils.draw_violation_alert(processed_frame)

            # Draw FPS counters
            yolo_fps = self._yolo_fps_counter.tick() if run_yolo else None
            processed_frame = self._draw_fps(processed_frame, display_fps, yolo_fps)

            if self.moving_camera_mode:
                self.infraction_recorder.add_frame(processed_frame)
                if current_violation:
                    violating_ids = [
                        vid for vid, vdata in self.tracker.vehicles.items()
                        if vdata.get("in_zone", False)
                    ]
                    for vid in violating_ids:
                        session = self.infraction_recorder.trigger_violation(
                            processed_frame, vehicle_id=vid
                        )
                        if session and self.on_violation_saved:
                            self.on_violation_saved(session)

            processed_frame = cv2.resize(
                processed_frame, (self.canvas_width, self.canvas_height),
                interpolation=cv2.INTER_LINEAR
            )
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)

            if not self.frame_queue.full():
                self.frame_queue.put(img)

        print(f"[Processor] Stopped after {frame_count} frames")

    def _handle_moving_camera(self, frame, frame_width, frame_height,
                               ego_state, polygon, calib_retry_counter):
        """Handles the state transitions and logic for moving camera mode, including detecting when the camera is stopped, running auto-calibration, and updating the crosswalk polygon. This method is called on every frame when in moving camera mode and manages the ego state machine."""
        is_stopped, motion = self.motion_detector.update(frame)

        if is_stopped:
            if ego_state == "Moving":
                ego_state = "Calibrating"
                self._push_status_frame(frame, ego_state)
                self._run_calibration_async(frame, frame_width, frame_height)

            elif ego_state == "Calibrating":
                with self._calib_lock:
                    result = self._calib_result
                    self._calib_result = None
                if result == "failed":
                    polygon = []
                    self.polygon_percent = []
                    ego_state = "Stopped (No Crosswalk)"
                    calib_retry_counter = 0
                elif result is not None:
                    self.polygon_percent = result
                    polygon = utils.get_crosswalk_polygon(
                        frame_width, frame_height, self.polygon_percent
                    )
                    ego_state = "Stopped & Monitoring"
                    calib_retry_counter = 0

            elif ego_state == "Stopped (No Crosswalk)":
                calib_retry_counter += 1
                if calib_retry_counter >= 30:
                    calib_retry_counter = 0
                    ego_state = "Calibrating"
                    self._push_status_frame(frame, ego_state)
                    self._run_calibration_async(frame, frame_width, frame_height)
        else:
            if ego_state != "Moving":
                ego_state = "Moving"
                polygon = []
                self.polygon_percent = []
                self.tracker = utils.VehicleTracker(
                    max_distance=100, max_frames_missing=30
                )

        return ego_state, polygon, calib_retry_counter

    def _run_calibration_async(self, frame, frame_width, frame_height):
        """Runs the auto-calibration process in a separate thread to avoid blocking the main video processing loop. The result of the calibration is stored in a thread-safe manner and will be applied in the main loop once it's ready."""
        with self._calib_lock:
            if self._calib_running:
                return
            self._calib_running = True
            self._calib_result = None

        def _do_calib():
            """Performs the auto-calibration process and stores the result. This function runs in a separate thread to avoid blocking the main video processing loop."""
            poly = self.auto_calibrator.calibrate(frame)
            with self._calib_lock:
                if poly is not None:
                    self._calib_result = [
                        (int(pt[0] / frame_width * 100),
                         int(pt[1] / frame_height * 100))
                        for pt in poly
                    ]
                else:
                    self._calib_result = "failed"
                self._calib_running = False

        threading.Thread(target=_do_calib, daemon=True, name="AutoCalib").start()

    def _draw_fps(self, frame, display_fps, yolo_fps):
        """Draws the FPS counters for both the display and YOLO processing on the given frame."""
        h, w = frame.shape[:2]
        cv2.putText(
            frame, f"Display: {display_fps:.1f} FPS",
            (w - 220, 30), config.FONT, 0.6, config.COLOR_WHITE, 1
        )
        if yolo_fps is not None:
            cv2.putText(
                frame, f"YOLO: {yolo_fps:.1f} FPS",
                (w - 220, 55), config.FONT, 0.6, config.COLOR_YELLOW, 1
            )
        return frame

    def _push_status_frame(self, frame, ego_state):
        """Pushes a frame with the current ego state drawn on it to the display queue. This is used to immediately update the UI with the current status when transitioning between states in moving camera mode, such as when starting calibration."""
        status_frame = frame.copy()
        status_frame = utils.draw_status_panel(
            status_frame, 0, 0, False, self.violation_count,
            ego_status=ego_state
        )
        resized = cv2.resize(
            status_frame, (self.canvas_width, self.canvas_height),
            interpolation=cv2.INTER_LINEAR
        )
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
        if not self.frame_queue.full():
            self.frame_queue.put(img)


class SmartCrosswalkApp:
    """Main application class that sets up the UI, manages state, and coordinates between the video capture, live preview, and video processing threads. Handles user interactions for opening video sources, starting/stopping detection, calibration, and moving camera mode. Also manages the display queue for updating the UI with processed frames and handles stream disconnection events."""
    def __init__(self, window):
        self.window = window
        self.window.title("SafeWalk AI - Smart Crosswalk Monitor")
        self.window.geometry("1200x800")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.display_queue = Queue(maxsize=5)
        self.model = None
        self.settings_file = "settings.json"
        self.points = []
        self.mode = "IDLE"
        self.polygon_percent = []
        self.capture_manager = CaptureManager()
        self.violation_log = []

        self._preview_stop = threading.Event()
        self._preview_thread = None
        self._processor_stop = threading.Event()
        self._processor_thread = None

        self.setup_ui()
        self.process_queue()
        self.check_stream_health()

    def save_settings(self):
        """Persists the current crosswalk polygon settings to a JSON file. This allows the application to remember the calibration settings between sessions. The polygon is saved as a list of percentage coordinates relative to the frame size, which allows it to be resolution-independent when reloaded."""
        with open(self.settings_file, "w") as f:
            json.dump({"polygon": self.polygon_percent}, f)

    def setup_ui(self):
        """Sets up the user interface using Tkinter. This includes the control panel on the left with buttons for opening video sources, starting/stopping detection, calibration options, and a canvas on the right for displaying the video feed with overlays. The control panel also includes labels for showing violation counts and stream health status, as well as a status label at the bottom for general messages."""
        self.controls = ttk.Frame(self.window, padding="10")
        self.controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            self.controls, text="SafeWalk AI", font=("Arial", 14, "bold")
        ).pack(pady=10)

        self.btn_open = ttk.Button(
            self.controls, text="Open video / camera", command=self.open_source
        )
        self.btn_open.pack(fill=tk.X, pady=5)

        self.btn_calib = ttk.Button(
            self.controls, text="Manual Calibration", command=self.start_calibration
        )
        self.btn_calib.pack(fill=tk.X, pady=5)

        self.btn_auto_calib = ttk.Button(
            self.controls, text="Auto Calibration", command=self.start_auto_calibration
        )
        self.btn_auto_calib.pack(fill=tk.X, pady=5)

        self.moving_camera_var = tk.BooleanVar(value=False)
        self.chk_moving = ttk.Checkbutton(
            self.controls, text="Moving Camera Mode",
            variable=self.moving_camera_var,
            command=self.toggle_moving_camera_mode
        )
        self.chk_moving.pack(fill=tk.X, pady=10)

        self.btn_run = ttk.Button(
            self.controls, text="Start Detection", command=self.start_detection
        )
        self.btn_run.pack(fill=tk.X, pady=5)

        self.btn_stop = ttk.Button(
            self.controls, text="Stop Detection", command=self.stop_detection
        )
        self.btn_stop.pack(fill=tk.X, pady=5)

        ttk.Separator(self.controls, orient="horizontal").pack(fill=tk.X, pady=10)

        self.btn_recordings = ttk.Button(
            self.controls, text="View Recordings",
            command=self.open_recordings_folder
        )
        self.btn_recordings.pack(fill=tk.X, pady=5)

        self.violation_count_label = ttk.Label(
            self.controls, text="Violations: 0", font=("Arial", 11, "bold"),
            foreground="red"
        )
        self.violation_count_label.pack(pady=5)

        self.stream_health_label = ttk.Label(
            self.controls, text="", foreground="gray"
        )
        self.stream_health_label.pack(pady=2)

        self.status_label = ttk.Label(
            self.controls, text="Status: Ready", foreground="blue",
            wraplength=200
        )
        self.status_label.pack(side=tk.BOTTOM, pady=20)

        self.CANVAS_W, self.CANVAS_H = 960, 540
        self.canvas = tk.Canvas(
            self.window, bg="black", width=self.CANVAS_W, height=self.CANVAS_H
        )
        self.canvas.pack(side=tk.RIGHT, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)

    def toggle_moving_camera_mode(self):
        """Enables or disables moving camera mode based on the state of the checkbox. When moving camera mode is enabled, the manual and auto calibration buttons are disabled since calibration is handled automatically in this mode. When moving camera mode is disabled, the calibration buttons are re-enabled to allow manual calibration."""
        if self.moving_camera_var.get():
            self.btn_calib.config(state=tk.DISABLED)
            self.btn_auto_calib.config(state=tk.DISABLED)
        else:
            self.btn_calib.config(state=tk.NORMAL)
            self.btn_auto_calib.config(state=tk.NORMAL)

    def _start_preview_thread(self):
        """Starts the live preview thread that continuously captures frames from the video source and updates the preview display. If a preview thread is already running, it is stopped before starting a new one. The new preview thread is configured with the current capture manager, display queue, and polygon settings."""
        self._stop_preview_thread()
        self._preview_stop.clear()
        preview = LivePreviewThread(
            capture_manager=self.capture_manager,
            display_queue=self.display_queue,
            stop_event=self._preview_stop,
            canvas_width=self.CANVAS_W,
            canvas_height=self.CANVAS_H,
        )
        preview.set_polygon(self.polygon_percent)
        self._preview_thread = preview
        self._preview_thread.start()

    def _stop_preview_thread(self):
        """Stops the live preview thread if it is running. Sets the stop event to signal the thread to exit, then waits for the thread to finish with a timeout. After stopping, the thread reference is cleared and the stop event is reset for future use."""
        self._preview_stop.set()
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)
        self._preview_thread = None
        self._preview_stop.clear()

    def _stop_processor_thread(self):
        """Stops the video processing thread if it is running. Sets the stop event to signal the thread to exit, then waits for the thread to finish with a timeout. After stopping, the thread reference is cleared and the stop event is reset for future use."""
        self._processor_stop.set()
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=2.0)
        self._processor_thread = None
        self._processor_stop.clear()

    def _flush_display_queue(self):
        """Empties the display queue to remove any pending frames that may be outdated. This is useful when stopping detection or when a stream disconnects, to ensure that old frames are not displayed after the state has changed."""
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except Empty:
                break

    def _on_stream_disconnect(self):
        """Handles the event when the video stream is disconnected. This method is called from the video processing thread when it detects that the stream has been lost. It schedules a call to _handle_disconnect_on_main_thread on the main UI thread to safely update the UI and stop the processing thread."""
        self.window.after(0, self._handle_disconnect_on_main_thread)

    def _handle_disconnect_on_main_thread(self):
        """Handles the stream disconnection event on the main UI thread. If the current mode is RUNNING, it stops the processor thread, flushes the display queue, updates the mode to IDLE, and shows a warning message to the user indicating that the stream was lost and they should check their phone and reconnect."""
        if self.mode == "RUNNING":
            self._stop_processor_thread()
            self._flush_display_queue()
            self.mode = "IDLE"
            self.status_label.config(
                text="Status: Stream disconnected. Reconnect your phone."
            )
            messagebox.showwarning(
                "Stream Lost",
                "The DroidCam stream was lost.\n"
                "Please check your phone and reconnect."
            )

    def _on_violation_saved(self, session_id: str):
        """Handles the event when a violation is saved. This method is called from the video processing thread when a new violation infraction is recorded. It appends the violation information to the violation log and updates the violation count label on the UI to reflect the total number of violations recorded."""
        self.violation_log.append({
            "session": session_id,
            "time": time.strftime("%H:%M:%S")
        })
        total = len(self.violation_log)
        self.window.after(
            0,
            lambda: self.violation_count_label.config(
                text=f"Violations recorded: {total}"
            )
        )

    def check_stream_health(self):
        """Periodically checks the health of the video stream and updates the stream health label on the UI. If the stream is live and open, it shows the current frame count. If the stream is live but not open, it shows a disconnected message. If there is no live stream, it clears the health label. This method reschedules itself to run every second to continuously monitor the stream status."""
        if self.capture_manager.is_live and self.capture_manager.is_open:
            reader = self.capture_manager._mjpeg_reader
            if reader:
                fc = reader.frame_count
                self.stream_health_label.config(
                    text=f"Stream frames: {fc}", foreground="green"
                )
        elif self.capture_manager.is_live and not self.capture_manager.is_open:
            self.stream_health_label.config(
                text="Stream: disconnected", foreground="red"
            )
        else:
            self.stream_health_label.config(text="")
        self.window.after(1000, self.check_stream_health)

    def open_recordings_folder(self):
        """Opens the folder where infraction recordings are saved. This method is called when the user clicks the "View Recordings" button. It ensures that the folder exists and then opens it using the default file explorer on the user's operating system."""
        folder = os.path.abspath(config.INFRACTIONS_DIR)
        os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')

    def open_source(self):
        """Handles the user action to open a video source. Prompts the user to choose between using a live stream from DroidCam or selecting a local video file. Depending on the choice, it either connects to the DroidCam stream using the provided IP address or opens the selected video file. After successfully opening the source, it starts the live preview thread and updates the status label accordingly. If there are any errors during connection or file opening, it shows an error message to the user."""
        from tkinter import simpledialog

        choice = messagebox.askyesnocancel(
            "Video Source",
            "Do you want to use a DroidCam live stream?\n\n"
            "(Click 'Yes' for DroidCam, 'No' for an MP4 file)"
        )

        if choice is True:
            ip_address = simpledialog.askstring(
                "DroidCam IP Address",
                "Enter the Wi-Fi IP address shown on your iPhone:",
                initialvalue="192.168.1."
            )
            if not ip_address:
                return

            ip = ip_address.strip()
            urls_to_try = [
                f"http://{ip}:4747/video",
                f"http://{ip}:4747/mjpegfeed",
                f"http://{ip}:4747/mjpeg.html",
            ]

            self.status_label.config(text="Status: Connecting to DroidCam...")
            self.window.update()

            connected = False
            for url in urls_to_try:
                print(f"[App] Trying: {url}")
                ret, frame = self.capture_manager.open(url, is_live=True)
                if ret and frame is not None:
                    connected = True
                    print(f"[App] Connected: {url}")
                    break

            if connected:
                self.polygon_percent = []
                self._flush_display_queue()
                self._stop_processor_thread()
                self._start_preview_thread()
                self.mode = "PREVIEW"
                self.status_label.config(
                    text="Status: Live preview — calibrate then start detection"
                )
            else:
                messagebox.showerror(
                    "Connection Error",
                    "Could not connect to DroidCam.\n\n"
                    "Check:\n"
                    "  - IP address is correct\n"
                    "  - DroidCam is open on your iPhone\n"
                    "  - PC and iPhone are on the same Wi-Fi\n"
                    "  - Port 4747 is not blocked by a firewall\n\n"
                    "URLs tried:\n" + "\n".join(f"  {u}" for u in urls_to_try)
                )

        elif choice is False:
            path = filedialog.askopenfilename(
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            if not path:
                return

            self.status_label.config(text="Status: Loading video...")
            self.window.update()

            ret, frame = self.capture_manager.open(path, is_live=False)
            if ret and frame is not None:
                self.polygon_percent = []
                self._flush_display_queue()
                self._stop_processor_thread()
                self._start_preview_thread()
                self.mode = "PREVIEW"
                self.status_label.config(
                    text="Status: Video loaded — calibrate then start detection"
                )
            else:
                messagebox.showerror("Error", "Could not read the video file.")

    def get_auto_polygon(self, frame):
        """Runs the auto-calibration process on the given frame to detect the crosswalk polygon. This method temporarily modifies the Python path to import the auto-calibration utilities without causing conflicts with the main application's modules. It loads the pre-trained model, predicts the trimap for the input frame, and extracts the polygon coordinates. The resulting polygon is returned as a list of points, or None if no valid polygon is detected."""
        auto_calib_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'auto_calibrate')
        )
        root_config = sys.modules.get('config')
        root_utils = sys.modules.get('utils')
        if 'config' in sys.modules: del sys.modules['config']
        if 'utils' in sys.modules: del sys.modules['utils']
        sys.path.insert(0, auto_calib_dir)
        try:
            from poly_utils import load_model, predict_trimap, extract_polygon
            import config as ac_config
            model = load_model(ac_config.FINETUNE_CKPT)
            trimap = predict_trimap(model, frame)
            return extract_polygon(trimap)
        finally:
            sys.path.pop(0)
            if root_config: sys.modules['config'] = root_config
            if root_utils: sys.modules['utils'] = root_utils

    def start_auto_calibration(self):
        """Initiates the AI auto-calibration process to automatically detect the crosswalk polygon from the current video frame. This method checks if a video source is loaded, then captures a snapshot frame and runs the auto-calibration algorithm. If a valid polygon is detected, it updates the polygon settings and applies them to the preview and processing threads. The user is notified of the success or failure of the auto-calibration process through status messages and dialog boxes."""
        if not self.capture_manager.is_open:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        self.status_label.config(text="Status: AI Auto-Calibration in progress...")
        self.window.update()

        ret, frame = self.capture_manager.get_snapshot()
        if not ret or frame is None:
            messagebox.showerror("Error", "Unable to read video frame.")
            return

        try:
            poly = self.get_auto_polygon(frame)
            if poly is not None:
                h, w = frame.shape[:2]
                self.polygon_percent = [
                    (int(pt[0] / w * 100), int(pt[1] / h * 100))
                    for pt in poly
                ]
                self.save_settings()
                if self._preview_thread is not None:
                    self._preview_thread.set_polygon(self.polygon_percent)
                self.status_label.config(
                    text="Status: Auto-Calibration done — ready to start detection"
                )
                messagebox.showinfo("Success", "Auto-calibration completed!")
            else:
                self.status_label.config(text="Status: Auto-Calibration failed")
                messagebox.showerror(
                    "Error", "No pedestrian crossing detected by the AI."
                )
        except Exception as e:
            self.status_label.config(text="Status: Auto-Calibration error")
            messagebox.showerror("Error", f"Auto-calibration error:\n{str(e)}")

    def start_calibration(self):
        """Starts the manual calibration process, allowing the user to click on the video preview to define the crosswalk zone. This method checks if a video source is loaded, then switches the application mode to CALIBRATING and prompts the user to click 4 points on the canvas to define the polygon. The clicked points are collected and displayed on the canvas, and once 4 points are defined, they are converted to percentage coordinates, saved to settings, and applied to the preview and processing threads. The user is notified of the calibration status through status messages and dialog boxes."""
        if not self.capture_manager.is_open:
            messagebox.showwarning("Warning", "Please load a video first.")
            return
        self.mode = "CALIBRATING"
        self.points = []
        self.canvas.delete("calibration")
        self.status_label.config(
            text="Calibration — click 4 points to define the crosswalk zone"
        )

    def on_canvas_click(self, event):
        """Handles mouse click events on the canvas during calibration mode. When the user clicks on the canvas, this method records the click coordinates as points for defining the crosswalk polygon. It visually marks the clicked points and draws a temporary polygon as the user clicks. Once 4 points are defined, it converts them to percentage coordinates, saves the settings, updates the preview and processing threads with the new polygon, and switches back to PREVIEW mode."""
        if self.mode != "CALIBRATING":
            return

        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(
            x - 5, y - 5, x + 5, y + 5,
            fill="red", outline="white", tags="calibration"
        )

        if len(self.points) > 1:
            self.canvas.delete("temp_poly")
            self.canvas.create_polygon(
                self.points, fill="cyan", stipple="gray25",
                outline="cyan", tags=("calibration", "temp_poly")
            )

        if len(self.points) == 4:
            self.polygon_percent = []
            for px, py in self.points:
                self.polygon_percent.append((
                    int((px / self.CANVAS_W) * 100),
                    int((py / self.CANVAS_H) * 100)
                ))
            self.save_settings()
            self.canvas.delete("calibration")
            if self._preview_thread is not None:
                self._preview_thread.set_polygon(self.polygon_percent)
            self.mode = "PREVIEW"
            messagebox.showinfo("Calibration", "Zone saved successfully!")
            self.status_label.config(
                text="Status: Calibration done — ready to start detection"
            )

    def start_detection(self):
        """Starts the video processing thread to perform pedestrian and vehicle detection, as well as violation monitoring. This method checks if a video source is loaded and if calibration is done (if not in moving camera mode) before starting. It loads the AI model if not already loaded, resets the capture to the start, clears any previous violations, and starts the VideoProcessorThread with the current settings. The application mode is set to RUNNING and the status label is updated to indicate that detection is in progress."""
        if not self.capture_manager.is_open:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        moving_cam = self.moving_camera_var.get()
        if not moving_cam:
            if not self.polygon_percent or len(self.polygon_percent) < 3:
                messagebox.showwarning(
                    "Calibration required",
                    "Please calibrate the zone before starting detection."
                )
                return

        if self.mode == "RUNNING":
            return

        self.status_label.config(text="Status: Loading AI model...")
        self.window.update()

        if not self.model:
            self.model = YOLO(config.YOLO_MODEL)

        self._stop_preview_thread()
        self._flush_display_queue()
        self.capture_manager.reset_to_start()
        self.violation_log = []
        self.violation_count_label.config(text="Violations recorded: 0")

        self._processor_stop.clear()
        self._processor_thread = VideoProcessorThread(
            capture_manager=self.capture_manager,
            model=self.model,
            polygon_percent=self.polygon_percent,
            frame_queue=self.display_queue,
            stop_event=self._processor_stop,
            canvas_width=self.CANVAS_W,
            canvas_height=self.CANVAS_H,
            moving_camera_mode=moving_cam,
            on_disconnect=self._on_stream_disconnect,
            on_violation_saved=self._on_violation_saved,
        )
        self._processor_thread.start()
        self.mode = "RUNNING"
        self.status_label.config(text="Status: Detection in progress")

    def stop_detection(self):
        """Stops the video processing thread if it is currently running. This method checks if the application mode is RUNNING before attempting to stop the processor thread. It signals the processor thread to stop, waits for it to finish, flushes the display queue to clear any pending frames, and then checks if the video source is still open. If the source is open, it restarts the live preview thread and updates the mode to PREVIEW with an appropriate status message. If the source is not open, it sets the mode to IDLE and updates the status to indicate that detection has stopped."""
        if self.mode != "RUNNING":
            return
        self._stop_processor_thread()
        self._flush_display_queue()
        if self.capture_manager.is_open:
            self._start_preview_thread()
            self.mode = "PREVIEW"
            self.status_label.config(
                text="Status: Detection stopped — live preview active"
            )
        else:
            self.mode = "IDLE"
            self.status_label.config(text="Status: Stopped")

    def process_queue(self):
        """Continuously processes the display queue to update the video feed on the canvas. This method is scheduled to run every 10 milliseconds using the Tkinter after method. If the application mode is PREVIEW, CALIBRATING, or RUNNING, it attempts to get a frame from the display queue without blocking. If a frame is available, it converts it to a PhotoImage and updates the canvas image. If the queue is empty, it simply passes and waits for the next scheduled call."""
        if self.mode in ("PREVIEW", "CALIBRATING", "RUNNING"):
            try:
                img = self.display_queue.get_nowait()
                self.photo = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.itemconfig(self.canvas_image, image=self.photo)
            except Empty:
                pass
        self.window.after(10, self.process_queue)

    def stop_all(self):
        """Stops all running threads and closes the video capture. This method is called when the application is closing to ensure that all resources are properly released. It signals both the processor and preview threads to stop, waits for them to finish, closes the capture manager, and updates the application mode and status label accordingly."""
        self._stop_processor_thread()
        self._stop_preview_thread()
        self.capture_manager.close()
        self.mode = "IDLE"
        self.status_label.config(text="Status: Stopped")

    def on_closing(self):
        """Handles the application closing event. This method is called when the user attempts to close the application window. It first calls stop_all to ensure that all threads are stopped and resources are released, then it destroys the main window to exit the application."""
        self.stop_all()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCrosswalkApp(root)
    root.mainloop()
