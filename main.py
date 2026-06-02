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
from queue import Queue, Empty
from ultralytics import YOLO
import utils
import config

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
            
            auto_calib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'auto_calibrate'))
            
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

class VideoProcessorThread(threading.Thread):
    def __init__(self, source, model, polygon_percent, frame_queue, stop_event, canvas_width, canvas_height, moving_camera_mode=False):
        super().__init__()
        self.source = source
        self.model = model
        self.polygon_percent = polygon_percent
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.daemon = True
        self.moving_camera_mode = moving_camera_mode
        
        # Init
        self.tracker = utils.VehicleTracker(max_distance=100, max_frames_missing=30)
        self.violation_count = 0

    def run(self):
        cap = cv2.VideoCapture(self.source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        if self.moving_camera_mode:
            self.infraction_recorder = utils.InfractionRecorder(fps=fps)
            self.motion_detector = utils.CameraMotionDetector()
            self.auto_calibrator = AutoCalibrator()
            ego_state = "Moving"
            polygon = []
            self.polygon_percent = []
            calib_retry_counter = 0
        else:
            polygon = utils.get_crosswalk_polygon(frame_width, frame_height, self.polygon_percent)
            ego_state = None

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            
            if self.moving_camera_mode:
                is_stopped, motion = self.motion_detector.update(frame)
                
                if is_stopped:
                    if ego_state == "Moving":
                        ego_state = "Calibrating"
                        status_frame = frame.copy()
                        status_frame = utils.draw_status_panel(
                            status_frame, 
                            0, 0, False, self.violation_count,
                            ego_status=ego_state
                        )
                        resized_status = cv2.resize(status_frame, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_LINEAR)
                        img = cv2.cvtColor(resized_status, cv2.COLOR_BGR2RGB)
                        img = PIL.Image.fromarray(img)
                        if not self.frame_queue.full():
                            self.frame_queue.put(img)
                            
                        poly = self.auto_calibrator.calibrate(frame)
                        if poly is not None:
                            self.polygon_percent = [(int(pt[0]/frame_width*100), int(pt[1]/frame_height*100)) for pt in poly]
                            polygon = utils.get_crosswalk_polygon(frame_width, frame_height, self.polygon_percent)
                            ego_state = "Stopped & Monitoring"
                            calib_retry_counter = 0
                        else:
                            polygon = []
                            self.polygon_percent = []
                            ego_state = "Stopped (No Crosswalk)"
                            calib_retry_counter = 0
                    elif ego_state == "Stopped (No Crosswalk)":
                        calib_retry_counter += 1
                        if calib_retry_counter >= 30:  # Retry calibration every ~1 second (30 frames)
                            calib_retry_counter = 0
                            ego_state = "Calibrating"
                            status_frame = frame.copy()
                            status_frame = utils.draw_status_panel(
                                status_frame, 
                                0, 0, False, self.violation_count,
                                ego_status=ego_state
                            )
                            resized_status = cv2.resize(status_frame, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_LINEAR)
                            img = cv2.cvtColor(resized_status, cv2.COLOR_BGR2RGB)
                            img = PIL.Image.fromarray(img)
                            if not self.frame_queue.full():
                                self.frame_queue.put(img)
                                
                            poly = self.auto_calibrator.calibrate(frame)
                            if poly is not None:
                                self.polygon_percent = [(int(pt[0]/frame_width*100), int(pt[1]/frame_height*100)) for pt in poly]
                                polygon = utils.get_crosswalk_polygon(frame_width, frame_height, self.polygon_percent)
                                ego_state = "Stopped & Monitoring"
                            else:
                                ego_state = "Stopped (No Crosswalk)"
                else:
                    if ego_state != "Moving":
                        ego_state = "Moving"
                        polygon = []
                        self.polygon_percent = []
                        self.tracker = utils.VehicleTracker(max_distance=100, max_frames_missing=30)
            
            # Track vehicles + detection of violation
            monitoring_active = (not self.moving_camera_mode) or (ego_state == "Stopped & Monitoring")
            
            persons_in_zone = 0
            person_bboxes = []
            vehicle_bboxes = []
            vehicles_in_zone_bboxes = []
            
            if monitoring_active:
                # YOLO treatment
                results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
                
                # Sort detections
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    if cls_id == config.PERSON_CLASS_ID:
                        if len(polygon) >= 3 and utils.is_in_zone(bbox, polygon):
                            persons_in_zone += 1
                            person_bboxes.append((bbox, f"Ped {conf:.0%} [IN]", config.COLOR_YELLOW))
                        else:
                            person_bboxes.append((bbox, f"Ped {conf:.0%}", config.COLOR_GREEN))
                    elif cls_id in config.VEHICLE_CLASSES:
                        class_name = self.model.names[cls_id]
                        vehicle_bboxes.append((bbox, class_name, conf))

                if len(polygon) >= 3:
                    vehicle_detections = [v[0] for v in vehicle_bboxes]
                    if persons_in_zone > 0:
                        new_violations, vehicles_in_zone_bboxes = self.tracker.update(vehicle_detections, polygon)
                        self.violation_count += new_violations
                    else:
                        _, vehicles_in_zone_bboxes = self.tracker.update(vehicle_detections, polygon)

            current_violation = (monitoring_active and persons_in_zone > 0 and len(vehicles_in_zone_bboxes) > 0)
            
            # Draw elements
            processed_frame = frame.copy()
            
            for bbox, label, color in person_bboxes:
                processed_frame = utils.draw_detection(processed_frame, bbox, label, color)
                
            for bbox, class_name, conf in vehicle_bboxes:
                in_zone = monitoring_active and (bbox in vehicles_in_zone_bboxes or (len(polygon) >= 3 and utils.is_in_zone(bbox, polygon)))
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
                processed_frame = utils.draw_detection(processed_frame, bbox, label, color, is_violation=is_viol)

            if len(polygon) >= 3:
                processed_frame = utils.draw_crosswalk_polygon(processed_frame, polygon, current_violation)
                
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

            # If Moving Camera Mode, record infraction if active
            if self.moving_camera_mode:
                self.infraction_recorder.add_frame(processed_frame)
                if current_violation:
                    violating_ids = [vid for vid, vdata in self.tracker.vehicles.items() if vdata.get("in_zone", False)]
                    for vid in violating_ids:
                        self.infraction_recorder.trigger_violation(processed_frame, vehicle_id=vid)

            processed_frame = cv2.resize(processed_frame, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            
            if not self.frame_queue.full():
                self.frame_queue.put(img)

        cap.release()

class SmartCrosswalkApp:
    def __init__(self, window):
        self.window = window
        self.window.title("SafeWalk AI - Smart Crosswalk Monitor")
        self.window.geometry("1200x800")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.stop_event = threading.Event()
        self.frame_queue = Queue(maxsize=3)
        self.model = None
        self.settings_file = "settings.json"
        self.source = None
        self.points = []
        self.mode = "IDLE"
        self.current_frame_cv2 = None
        self.polygon_percent = [] 
        self.setup_ui()
        self.process_queue()

    def save_settings(self):
        with open(self.settings_file, "w") as f:
            json.dump({"polygon": self.polygon_percent}, f)

    def setup_ui(self):
        self.controls = ttk.Frame(self.window, padding="10")
        self.controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(self.controls, text="MENU", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.btn_open = ttk.Button(self.controls, text="Open video", command=self.open_source)
        self.btn_open.pack(fill=tk.X, pady=5)
        self.btn_calib = ttk.Button(self.controls, text="Manual Calibration", command=self.start_calibration)
        self.btn_calib.pack(fill=tk.X, pady=5)
        self.btn_auto_calib = ttk.Button(self.controls, text="Auto Calibration", command=self.start_auto_calibration)
        self.btn_auto_calib.pack(fill=tk.X, pady=5)
        
        self.moving_camera_var = tk.BooleanVar(value=False)
        self.chk_moving = ttk.Checkbutton(self.controls, text="Moving Camera Mode", variable=self.moving_camera_var, command=self.toggle_moving_camera_mode)
        self.chk_moving.pack(fill=tk.X, pady=10)
        
        self.btn_run = ttk.Button(self.controls, text="Start Detection", command=self.start_detection)
        self.btn_run.pack(fill=tk.X, pady=5)
        self.btn_stop = ttk.Button(self.controls, text="Stop", command=self.stop_all)
        self.btn_stop.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(self.controls, text="Status: Ready", foreground="blue")
        self.status_label.pack(side=tk.BOTTOM, pady=20)
        self.CANVAS_W, self.CANVAS_H = 960, 540
        self.canvas = tk.Canvas(self.window, bg="black", width=self.CANVAS_W, height=self.CANVAS_H)
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

    def display_first_frame(self, frame):
        self.current_frame_cv2 = frame
        
        frame_resized = cv2.resize(frame, (self.CANVAS_W, self.CANVAS_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Draw polygon if already calibrated
        if self.polygon_percent and len(self.polygon_percent) >= 3:
            poly_px = utils.get_crosswalk_polygon(self.CANVAS_W, self.CANVAS_H, self.polygon_percent)
            cv2.polylines(img, [np.array(poly_px, np.int32)], True, (0, 255, 0), 2)
            
        img = PIL.Image.fromarray(img)
        self.photo = PIL.ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)

    def open_source(self):
        path = filedialog.askopenfilename()
        if path:
            self.source = path
            self.polygon_percent = []
            self.status_label.config(text=f"Video loaded (Please calibrate)")
            cap = cv2.VideoCapture(self.source)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.display_first_frame(frame)

    def get_auto_polygon(self, frame):
        auto_calib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'auto_calibrate'))
        
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
        if not self.source:
            messagebox.showwarning("Attention", "Please load a video first.")
            return

        self.status_label.config(text="Status: IA Auto-Calib in progress...")
        self.window.update()

        cap = cv2.VideoCapture(self.source)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Unable to read the video.")
            return

        try:
            poly = self.get_auto_polygon(frame)
            if poly is not None:
                h, w = frame.shape[:2]
                self.polygon_percent = [(int(pt[0]/w*100), int(pt[1]/h*100)) for pt in poly]
                self.save_settings()
                self.status_label.config(text="Status: Auto-Calib Success")
                messagebox.showinfo("Success", "Auto-Calibration completed successfully!")
                self.display_first_frame(frame)
            else:
                self.status_label.config(text="Status: Auto-Calib Failed")
                messagebox.showerror("Error", "No pedestrian crossing detected by the AI.")
        except Exception as e:
            self.status_label.config(text="Status: AI Auto-Calib Error")
            messagebox.showerror("Error", f"Error during auto-calibration:\n{str(e)}")

    def start_calibration(self):
        if not self.source:
            messagebox.showwarning("Attention", "Please load a video first.")
            return
        self.mode = "CALIBRATING"
        self.points = []
        self.canvas.delete("calibration")
        self.status_label.config(text="Mode: Calibration - Click 4 points to define the crosswalk zone")

    def on_canvas_click(self, event):
        if self.mode != "CALIBRATING": return
        
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="white", tags="calibration")

        if len(self.points) > 1:
            self.canvas.delete("temp_poly")
            self.canvas.create_polygon(self.points, fill="cyan", stipple="gray25", outline="cyan", tags=("calibration", "temp_poly"))

        if len(self.points) == 4:
            self.polygon_percent = []
            for px, py in self.points:
                pc_x = int((px / self.CANVAS_W) * 100)
                pc_y = int((py / self.CANVAS_H) * 100)
                self.polygon_percent.append((pc_x, pc_y))
            
            self.save_settings()
            self.mode = "IDLE"
            messagebox.showinfo("Calibration", "Zone saved!")
            self.status_label.config(text="Status: Calibration completed")
            if self.current_frame_cv2 is not None:
                self.display_first_frame(self.current_frame_cv2)

    def start_detection(self):
        if not self.source:
            messagebox.showwarning("Attention", "Please load a video first.")
            return

        moving_cam = self.moving_camera_var.get()
        if not moving_cam:
            if not self.polygon_percent or len(self.polygon_percent) < 3:
                messagebox.showwarning("Calibration required", "Please calibrate the zone (Manual or Auto) before starting detection.")
                return
            
        if self.mode == "RUNNING": return
        
        self.status_label.config(text="Status: Loading AI...")
        self.window.update()
        
        if not self.model:
            self.model = YOLO(config.YOLO_MODEL)
            
        self.stop_all()
        
        self.processor_thread = VideoProcessorThread(
            self.source, 
            self.model, 
            self.polygon_percent, 
            self.frame_queue, 
            self.stop_event, 
            self.CANVAS_W, 
            self.CANVAS_H,
            moving_camera_mode=moving_cam
        )
        self.processor_thread.start()
        self.mode = "RUNNING"
        self.status_label.config(text="Status: Detection in progress")

    def process_queue(self):
        if self.mode == "RUNNING":
            try:
                img = self.frame_queue.get_nowait()
                self.photo = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.itemconfig(self.canvas_image, image=self.photo)
                
            except Empty:
                pass
        self.window.after(10, self.process_queue)

    def stop_all(self):
        self.stop_event.set()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=1.0)
            
        self.stop_event.clear()
        self.mode = "IDLE"
        self.status_label.config(text="Status: Stopped")
    
    def on_closing(self):
        self.stop_all()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCrosswalkApp(root)
    root.mainloop()
