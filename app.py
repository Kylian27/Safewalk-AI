import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import PIL.Image, PIL.ImageTk
import threading
import json
import os
import time
from queue import Queue, Empty
from ultralytics import YOLO
import numpy as np

# Importation de tes fonctions utilitaires
import utils
import config

class VideoProcessorThread(threading.Thread):
    def __init__(self, source, model, polygon_percent, frame_queue, stop_event, canvas_width, canvas_height):
        super().__init__()
        self.source = source
        self.model = model
        self.polygon_percent = polygon_percent
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.daemon = True # Le thread s'arrête si le main thread s'arrête

    def run(self):
        cap = cv2.VideoCapture(self.source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Récupération du polygone en pixels
        config.CROSSWALK_POLYGON_PERCENT = self.polygon_percent
        polygon = utils.get_crosswalk_polygon(frame_width, frame_height)
        
        violation_count = 0
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            
            # --- Traitement YOLO ---
            results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
            
            # 1. Dessiner les boîtes YOLO
            processed_frame = results[0].plot()
            
            # 2. Compter et Dessiner le polygone
            persons_in_zone = 0
            vehicles_in_zone = 0
            
            for box in results[0].boxes:
                cls = int(box.cls[0])
                bbox = box.xyxy[0].tolist()
                if cls == config.PERSON_CLASS_ID and utils.is_in_zone(bbox, polygon):
                    persons_in_zone += 1
                elif cls in config.VEHICLE_CLASSES and utils.is_in_zone(bbox, polygon):
                    vehicles_in_zone += 1

            violation = (persons_in_zone > 0 and vehicles_in_zone > 0)
            if violation: violation_count += 1
            
            # Dessiner le polygone rempli
            processed_frame = utils.draw_crosswalk_polygon(processed_frame, polygon, violation)
            
            # Dessiner le panneau de statut
            processed_frame = utils.draw_status_panel(processed_frame, persons_in_zone, vehicles_in_zone, violation, violation_count)
            
            # --- Préparation pour Tkinter (Thread-Safe) ---
            # Conversion BGR -> RGB
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            
            # Redimensionnement ICI pour décharger le GUI
            img = img.resize((self.canvas_width, self.canvas_height), PIL.Image.Resampling.LANCZOS)
            
            # Dépot dans la queue
            self.frame_queue.put(img)
            
            # Petit repos pour le CPU (50 FPS max)
            time.sleep(0.01)

        cap.release()

class SmartCrosswalkApp:
    def __init__(self, window):
        self.window = window
        self.window.title("SafeWalk AI - Smart Crosswalk Monitor")
        self.window.geometry("1200x800")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # --- État de l'application ---
        self.stop_event = threading.Event()
        self.frame_queue = Queue()
        self.model = None
        self.settings_file = "settings.json"
        self.source = None
        
        # Données de calibration
        self.points = []
        self.mode = "IDLE" # IDLE, CALIBRATING, RUNNING
        self.current_frame_cv2 = None
        self.polygon_percent = self.load_settings()

        self.setup_ui()
        self.process_queue() # Démarre la vérification de la queue

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, "r") as f:
                return json.load(f).get("polygon", config.CROSSWALK_POLYGON_PERCENT)
        return config.CROSSWALK_POLYGON_PERCENT

    def save_settings(self):
        with open(self.settings_file, "w") as f:
            json.dump({"polygon": self.polygon_percent}, f)

    def setup_ui(self):
        # Barre latérale de contrôle
        self.controls = ttk.Frame(self.window, padding="10")
        self.controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(self.controls, text="MENU", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.btn_open = ttk.Button(self.controls, text="📁 Open video", command=self.open_source)
        self.btn_open.pack(fill=tk.X, pady=5)

        self.btn_calib = ttk.Button(self.controls, text="📍 Calibrate Zone", command=self.start_calibration)
        self.btn_calib.pack(fill=tk.X, pady=5)

        self.btn_run = ttk.Button(self.controls, text="🚀 Start Detection", command=self.start_detection)
        self.btn_run.pack(fill=tk.X, pady=5)

        self.btn_stop = ttk.Button(self.controls, text="⏹ Stop", command=self.stop_all)
        self.btn_stop.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(self.controls, text="Status: Ready", foreground="blue")
        self.status_label.pack(side=tk.BOTTOM, pady=20)

        # Zone Vidéo (taille fixe pour optimiser)
        self.CANVAS_W, self.CANVAS_H = 960, 540
        self.canvas = tk.Canvas(self.window, bg="black", width=self.CANVAS_W, height=self.CANVAS_H)
        self.canvas.pack(side=tk.RIGHT, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def open_source(self):
        path = filedialog.askopenfilename()
        if path:
            self.source = path
            self.status_label.config(text=f"Video loaded")
            
            # Afficher la première frame pour calibration
            cap = cv2.VideoCapture(self.source)
            ret, frame = cap.read()
            if ret:
                self.current_frame_cv2 = frame
                # Conversion pour affichage initial
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(img)
                img = img.resize((self.CANVAS_W, self.CANVAS_H), PIL.Image.Resampling.LANCZOS)
                self.photo = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            cap.release()

    def start_calibration(self):
        if not self.source:
            messagebox.showwarning("Attention", "Veuillez charger une vidéo d'abord.")
            return
        self.mode = "CALIBRATING"
        self.points = []
        # Nettoyer les anciens polygones de calibration
        self.canvas.delete("calibration")
        self.status_label.config(text="Mode: Calibration (Cliquez 4 points)")

    def on_canvas_click(self, event):
        if self.mode != "CALIBRATING": return
        
        x, y = event.x, event.y
        self.points.append((x, y))
        
        # Dessiner le point
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="white", tags="calibration")

        # Dessiner le polygone en cours de remplissage
        if len(self.points) > 1:
            # On supprime le polygone temporaire précédent
            self.canvas.delete("temp_poly")
            
            # On dessine le polygone rempli mais semi-transparent (couleur cyan)
            self.canvas.create_polygon(self.points, fill="cyan", stipple="gray25", outline="cyan", tags=("calibration", "temp_poly"))

        if len(self.points) == 4:
            # Conversion en pourcentages
            self.polygon_percent = []
            for px, py in self.points:
                pc_x = int((px / self.CANVAS_W) * 100)
                pc_y = int((py / self.CANVAS_H) * 100)
                self.polygon_percent.append((pc_x, pc_y))
            
            self.save_settings()
            self.mode = "IDLE"
            messagebox.showinfo("Calibration", "Zone sauvegardée !")
            self.status_label.config(text="Statut: Calibré")

    def start_detection(self):
        if not self.source:
            messagebox.showwarning("Attention", "Veuillez charger une vidéo d'abord.")
            return
        if self.mode == "RUNNING": return
        
        self.status_label.config(text="Chargement IA...")
        
        if not self.model:
            self.model = YOLO(config.YOLO_MODEL)
            
        # Arrêter tout process en cours
        self.stop_all()
        
        # Création et démarrage du thread de traitement
        self.processor_thread = VideoProcessorThread(
            self.source, 
            self.model, 
            self.polygon_percent, 
            self.frame_queue, 
            self.stop_event, 
            self.CANVAS_W, 
            self.CANVAS_H
        )
        self.processor_thread.start()
        self.mode = "RUNNING"
        self.status_label.config(text="Statut: Détection en cours")

    def process_queue(self):
        """Vérifie la queue d'images régulièrement et met à jour le canvas."""
        if self.mode == "RUNNING":
            try:
                # Récupère une image de la queue
                img = self.frame_queue.get_nowait()
                
                # Affiche l'image
                self.photo = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
            except Empty:
                pass
        
        # Se relance automatiquement toutes les 10ms
        self.window.after(10, self.process_queue)

    def stop_all(self):
        """Arrête le thread de traitement."""
        self.stop_event.set()
        # On vide la queue pour débloquer le thread
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=1.0) # Attend proprement la fin
            
        self.stop_event.clear()
        self.mode = "IDLE"
        self.status_label.config(text="Statut: Arrêté")
    
    def on_closing(self):
        """Gère la fermeture complète de l'application."""
        print("Closing...")
        self.stop_all()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCrosswalkApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()