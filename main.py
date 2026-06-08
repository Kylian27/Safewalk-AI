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
    def __init__(self, url: str):
        self.url = url
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
        self._connected = False
        self._frame_count = 0

    def start(self) -> tuple[bool, object]:
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
        with self._lock:
            if self._latest_frame is None or not self._connected:
                return False, None
            return True, self._latest_frame.copy()

    def stop(self):
        self._running = False
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._lock:
            self._latest_frame = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _reader_loop(self, first_frame_event, first_frame_holder):
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
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None


class CaptureManager:
    def __init__(self):
        self.source = None
        self.is_live = False
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0
        self._mjpeg_reader = None
        self._file_cap = None

    def open(self, source: str, is_live: bool) -> tuple[bool, object]:
        self.close()
        self.source = source
        self.is_live = is_live
        if is_live:
            return self._open_live(source)
        else:
            return self._open_file(source)

    def get_frame(self) -> tuple[bool, object]:
        if self.is_live:
            if self._mjpeg_reader is None:
                return False, None
            return self._mjpeg_reader.get_frame()
        else:
            if self._file_cap is None or not self._file_cap.isOpened():
                return False, None
            return self._file_cap.read()

    def get_snapshot(self) -> tuple[bool, object]:
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
        if self._mjpeg_reader is not None:
            return self._mjpeg_reader.frame_count
        return 0

    def reset_to_start(self):
        if not self.is_live and self._file_cap is not None:
            self._file_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        if self._mjpeg_reader is not None:
            self._mjpeg_reader.stop()
            self._mjpeg_reader = None
        if self._file_cap is not None:
            self._file_cap.release()
            self._file_cap = None

    @property
    def is_open(self) -> bool:
        if self.is_live:
            return self._mjpeg_reader is not None and self._mjpeg_reader.is_connected
        else:
            return self._file_cap is not None and self._file_cap.isOpened()

    def _open_live(self, source: str) -> tuple[bool, object]:
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
    def __init__(self):
        self.model = None
        self.predict_trimap = None
        self.extract_polygon = None
        self.lock = threading.Lock()

    def load(self):
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
        self.load()
        if self.model is None:
            return None
        trimap = self.predict_trimap(self.model, frame)
        return self.extract_polygon(trimap)


class LivePreviewThread(threading.Thread):
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
        with self._polygon_lock:
            self.polygon_percent = list(polygon_percent)

    def run(self):
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
    """Tracks real-time FPS using a rolling window."""
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
        with self._calib_lock:
            if self._calib_running:
                return
            self._calib_running = True
            self._calib_result = None

        def _do_calib():
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
        with open(self.settings_file, "w") as f:
            json.dump({"polygon": self.polygon_percent}, f)

    def setup_ui(self):
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
        if self.moving_camera_var.get():
            self.btn_calib.config(state=tk.DISABLED)
            self.btn_auto_calib.config(state=tk.DISABLED)
        else:
            self.btn_calib.config(state=tk.NORMAL)
            self.btn_auto_calib.config(state=tk.NORMAL)

    def _start_preview_thread(self):
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
        self._preview_stop.set()
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)
        self._preview_thread = None
        self._preview_stop.clear()

    def _stop_processor_thread(self):
        self._processor_stop.set()
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=2.0)
        self._processor_thread = None
        self._processor_stop.clear()

    def _flush_display_queue(self):
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except Empty:
                break

    def _on_stream_disconnect(self):
        self.window.after(0, self._handle_disconnect_on_main_thread)

    def _handle_disconnect_on_main_thread(self):
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
        folder = os.path.abspath(config.INFRACTIONS_DIR)
        os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')

    def open_source(self):
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
        if self.mode in ("PREVIEW", "CALIBRATING", "RUNNING"):
            try:
                img = self.display_queue.get_nowait()
                self.photo = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.itemconfig(self.canvas_image, image=self.photo)
            except Empty:
                pass
        self.window.after(10, self.process_queue)

    def stop_all(self):
        self._stop_processor_thread()
        self._stop_preview_thread()
        self.capture_manager.close()
        self.mode = "IDLE"
        self.status_label.config(text="Status: Stopped")

    def on_closing(self):
        self.stop_all()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCrosswalkApp(root)
    root.mainloop()
