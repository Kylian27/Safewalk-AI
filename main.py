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

    def _reader_loop(self, first_frame_event: threading.Event, first_frame_holder: list):
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

    def _stream_jpeg_scan(self, response, first_frame_event, first_frame_holder, first_frame_sent):
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
                    print(f"[MJPEG] First frame decoded: {frame.shape}")
                if frames_decoded % 30 == 0:
                    print(f"[MJPEG] {self._frame_count} frames received")

    def _decode_jpeg(self, data: bytes):
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
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
        poly = self.extract_polygon(trimap)
        return poly


class LivePreviewThread(threading.Thread):
    """
    Continuously pulls frames from CaptureManager and pushes them
    to the display queue. Runs whenever a source is open.
    The app overlays calibration points on top via the canvas directly.
    """
    def __init__(self, capture_manager, display_queue, stop_event,
                 canvas_width, canvas_height):
        super().__init__(name="LivePreview", daemon=True)
        self.capture_manager = capture_manager
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def run(self):
        print("[Preview] Started")
        last_frame_count = -1

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

            frame_resized = cv2.resize(
                frame, (self.canvas_width, self.canvas_height),
                interpolation=cv2.INTER_LINEAR
            )
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)

            if not self.display_queue.full():
                self.display_queue.put(img)

        print("[Preview] Stopped")


class VideoProcessorThread(threading.Thread):
    """
    Runs YOLO detection on every new frame from CaptureManager.
    """
    def __init__(self, capture_manager, model, polygon_percent, frame_queue,
                 stop_event, canvas_width, canvas_height, moving_camera_mode=False):
        super().__init__(name="VideoProcessor", daemon=True)
        self.capture_manager = capture_manager
        self.model = model
        self.polygon_percent = polygon_percent
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.moving_camera_mode = moving_camera_mode
        self.tracker = utils.VehicleTracker(max_distance=100, max_frames_missing=30)
        self.violation_count = 0

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
        else:
            polygon = utils.get_crosswalk_polygon(
                frame_width, frame_height, self.polygon_percent
            )
            ego_state = None

        last_frame_count = -1
        frame_count = 0

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
                    time.sleep(0.033)
                    continue
                else:
                    print("[Processor] End of file")
                    break

            frame_count += 1

            if self.moving_camera_mode:
                is_stopped, motion = self.motion_detector.update(frame)

                if is_stopped:
                    if ego_state == "Moving":
                        ego_state = "Calibrating"
                        self._push_status_frame(frame, ego_state)
                        poly = self.auto_calibrator.calibrate(frame)
                        if poly is not None:
                            self.polygon_percent = [
                                (int(pt[0] / frame_width * 100),
                                 int(pt[1] / frame_height * 100))
                                for pt in poly
                            ]
                            polygon = utils.get_crosswalk_polygon(
                                frame_width, frame_height, self.polygon_percent
                            )
                            ego_state = "Stopped & Monitoring"
                            calib_retry_counter = 0
                        else:
                            polygon = []
                            self.polygon_percent = []
                            ego_state = "Stopped (No Crosswalk)"
                            calib_retry_counter = 0

                    elif ego_state == "Stopped (No Crosswalk)":
                        calib_retry_counter += 1
                        if calib_retry_counter >= 30:
                            calib_retry_counter = 0
                            ego_state = "Calibrating"
                            self._push_status_frame(frame, ego_state)
                            poly = self.auto_calibrator.calibrate(frame)
                            if poly is not None:
                                self.polygon_percent = [
                                    (int(pt[0] / frame_width * 100),
                                     int(pt[1] / frame_height * 100))
                                    for pt in poly
                                ]
                                polygon = utils.get_crosswalk_polygon(
                                    frame_width, frame_height, self.polygon_percent
                                )
                                ego_state = "Stopped & Monitoring"
                            else:
                                ego_state = "Stopped (No Crosswalk)"
                else:
                    if ego_state != "Moving":
                        ego_state = "Moving"
                        polygon = []
                        self.polygon_percent = []
                        self.tracker = utils.VehicleTracker(
                            max_distance=100, max_frames_missing=30
                        )

            monitoring_active = (
                not self.moving_camera_mode or ego_state == "Stopped & Monitoring"
            )

            persons_in_zone = 0
            person_bboxes = []
            vehicle_bboxes = []
            vehicles_in_zone_bboxes = []

            if monitoring_active:
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

            if self.moving_camera_mode:
                self.infraction_recorder.add_frame(processed_frame)
                if current_violation:
                    violating_ids = [
                        vid for vid, vdata in self.tracker.vehicles.items()
                        if vdata.get("in_zone", False)
                    ]
                    for vid in violating_ids:
                        self.infraction_recorder.trigger_violation(
                            processed_frame, vehicle_id=vid
                        )

            processed_frame = cv2.resize(
                processed_frame, (self.canvas_width, self.canvas_height),
                interpolation=cv2.INTER_LINEAR
            )
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)

            if not self.frame_queue.full():
                self.frame_queue.put(img)

        print(f"[Processor] Stopped after {frame_count} frames")

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

        # IDLE       -> nothing open
        # PREVIEW    -> live feed shown, no YOLO
        # CALIBRATING-> live feed shown, user clicking points (still PREVIEW underneath)
        # RUNNING    -> YOLO detection active
        self.mode = "IDLE"
        self.polygon_percent = []
        self.capture_manager = CaptureManager()

        self._preview_stop = threading.Event()
        self._preview_thread = None
        self._processor_stop = threading.Event()
        self._processor_thread = None

        self.setup_ui()
        self.process_queue()

    def save_settings(self):
        with open(self.settings_file, "w") as f:
            json.dump({"polygon": self.polygon_percent}, f)

    def setup_ui(self):
        self.controls = ttk.Frame(self.window, padding="10")
        self.controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            self.controls, text="MENU", font=("Arial", 14, "bold")
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

        self.status_label = ttk.Label(
            self.controls, text="Status: Ready", foreground="blue"
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
        self._preview_thread = LivePreviewThread(
            capture_manager=self.capture_manager,
            display_queue=self.display_queue,
            stop_event=self._preview_stop,
            canvas_width=self.CANVAS_W,
            canvas_height=self.CANVAS_H,
        )
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
            poly = extract_polygon(trimap)
            return poly
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
                self.status_label.config(
                    text="Status: Auto-Calibration done — ready to start detection"
                )
                messagebox.showinfo("Success", "Auto-calibration completed successfully!")
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
            self.status_label.config(text="Status: Detection stopped — live preview active")
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
