"""
Smart Crosswalk Monitor
Détecte les violations quand un véhicule ne cède pas le passage
à un piéton sur un passage piéton.

Usage:
    python main.py                          # Webcam
    python main.py --source videos/test.mp4 # Vidéo
    python main.py --source videos/test.mp4 --save  # Sauvegarder
"""

import cv2
import argparse
import time
from ultralytics import YOLO
from config import *
from utils import (
    get_crosswalk_polygon,
    draw_crosswalk_polygon,
    is_in_zone,
    draw_detection,
    draw_status_panel,
    draw_violation_alert
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Crosswalk Monitor")
    parser.add_argument("--source", type=str, default="0",
                        help="Chemin vidéo ou 0 pour webcam")
    parser.add_argument("--save", action="store_true",
                        help="Sauvegarder la vidéo de sortie")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Afficher la vidéo en temps réel")
    parser.add_argument("--model", type=str, default=YOLO_MODEL,
                        help="Modèle YOLO à utiliser")
    return parser.parse_args()


def get_box_center(bbox):
    """Retourne le centre d'une bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(p1, p2):
    """Distance euclidienne entre deux points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


class VehicleTracker:
    """
    Tracker simple pour suivre les véhicules et éviter 
    de compter plusieurs fois la même violation.
    """
    
    def __init__(self, max_distance=100, max_frames_missing=30):
        self.vehicles = {}  # id -> {"center": (x,y), "violated": bool, "missing": int}
        self.next_id = 0
        self.max_distance = max_distance  # Distance max pour matcher
        self.max_frames_missing = max_frames_missing  # Frames avant suppression
    
    def update(self, detections, polygon):
        """
        Met à jour le tracker avec les nouvelles détections.
        
        Args:
            detections: liste de bboxes [(x1,y1,x2,y2), ...]
            polygon: zone du passage piéton
            
        Returns:
            new_violations: nombre de NOUVELLES violations ce frame
            vehicles_in_zone: liste des véhicules actuellement en zone
        """
        new_violations = 0
        vehicles_in_zone = []
        matched_ids = set()
        
        for bbox in detections:
            center = get_box_center(bbox)
            in_zone = is_in_zone(bbox, polygon)
            
            # Chercher le véhicule le plus proche
            best_id = None
            best_dist = float('inf')
            
            for vid, vdata in self.vehicles.items():
                if vid in matched_ids:
                    continue
                d = distance(center, vdata["center"])
                if d < best_dist and d < self.max_distance:
                    best_dist = d
                    best_id = vid
            
            if best_id is not None:
                # Véhicule existant trouvé
                matched_ids.add(best_id)
                self.vehicles[best_id]["center"] = center
                self.vehicles[best_id]["missing"] = 0
                self.vehicles[best_id]["bbox"] = bbox
                
                # Vérifier si nouvelle violation
                if in_zone and not self.vehicles[best_id]["violated"]:
                    self.vehicles[best_id]["violated"] = True
                    self.vehicles[best_id]["in_zone"] = True
                    new_violations += 1
                elif in_zone:
                    self.vehicles[best_id]["in_zone"] = True
                else:
                    self.vehicles[best_id]["in_zone"] = False
                    
            else:
                # Nouveau véhicule
                self.vehicles[self.next_id] = {
                    "center": center,
                    "bbox": bbox,
                    "violated": False,
                    "in_zone": in_zone,
                    "missing": 0
                }
                
                # Si déjà dans la zone, c'est une violation
                if in_zone:
                    self.vehicles[self.next_id]["violated"] = True
                    new_violations += 1
                
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Incrémenter le compteur missing pour les véhicules non matchés
        ids_to_remove = []
        for vid in self.vehicles:
            if vid not in matched_ids:
                self.vehicles[vid]["missing"] += 1
                self.vehicles[vid]["in_zone"] = False
                if self.vehicles[vid]["missing"] > self.max_frames_missing:
                    ids_to_remove.append(vid)
        
        # Supprimer les véhicules disparus depuis trop longtemps
        for vid in ids_to_remove:
            del self.vehicles[vid]
        
        # Lister les véhicules actuellement dans la zone
        for vid, vdata in self.vehicles.items():
            if vdata.get("in_zone", False):
                vehicles_in_zone.append(vdata["bbox"])
        
        return new_violations, vehicles_in_zone
    
    def get_active_violations(self):
        """Retourne le nombre de véhicules actuellement en zone qui ont violé."""
        count = 0
        for vdata in self.vehicles.values():
            if vdata.get("in_zone", False) and vdata.get("violated", False):
                count += 1
        return count


def main():
    args = parse_args()
    
    # ================================
    # 1. Charger le modèle YOLO
    # ================================
    print("[INFO] Chargement du modèle YOLO...")
    model = YOLO(args.model)
    print(f"[INFO] Modèle {args.model} chargé avec succès!")
    
    # ================================
    # 2. Ouvrir la source vidéo
    # ================================
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la source: {args.source}")
        return
    
    # Propriétés de la vidéo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"[INFO] Vidéo: {frame_width}x{frame_height} @ {fps}fps")
    
    # ================================
    # 3. Calculer la zone passage piéton
    # ================================
    crosswalk_polygon = get_crosswalk_polygon(frame_width, frame_height)
    print(f"[INFO] Zone passage piéton: {crosswalk_polygon}")
    
    # ================================
    # 4. Préparer la sauvegarde vidéo
    # ================================
    writer = None
    if args.save:
        output_path = f"output/output_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                  (frame_width, frame_height))
        print(f"[INFO] Sauvegarde vers: {output_path}")
    
    # ================================
    # 5. Variables de suivi
    # ================================
    violation_count = 0
    frame_count = 0
    vehicle_tracker = VehicleTracker(
        max_distance=100,      # Distance max pour associer un véhicule
        max_frames_missing=30  # Frames avant d'oublier un véhicule
    )
    
    # ================================
    # 6. Boucle principale
    # ================================
    print("[INFO] Démarrage de la détection... (Appuie 'q' pour quitter)")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la vidéo.")
            break
        
        frame_count += 1
        
        # ----- DÉTECTION YOLO -----
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # ----- ANALYSE DES DÉTECTIONS -----
        persons_in_zone = 0
        person_bboxes = []
        vehicle_bboxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                # --- PIÉTONS ---
                if cls_id == PERSON_CLASS_ID:
                    in_zone = is_in_zone(bbox, crosswalk_polygon)
                    
                    if in_zone:
                        persons_in_zone += 1
                        label = f"Pedestrian {conf:.0%} [IN ZONE]"
                        color = COLOR_YELLOW
                    else:
                        label = f"Pedestrian {conf:.0%}"
                        color = COLOR_GREEN
                    
                    person_bboxes.append((bbox, label, color))
                
                # --- VÉHICULES ---
                elif cls_id in VEHICLE_CLASSES:
                    class_name = model.names[cls_id]
                    vehicle_bboxes.append((bbox, class_name, conf))
        
        # ----- TRACKING DES VÉHICULES -----
        # On ne track que s'il y a des piétons dans la zone
        vehicle_detections = [v[0] for v in vehicle_bboxes]
        
        if persons_in_zone > 0:
            new_violations, vehicles_in_zone_bboxes = vehicle_tracker.update(
                vehicle_detections, crosswalk_polygon
            )
            violation_count += new_violations
            
            if new_violations > 0:
                print(f"[⚠ VIOLATION #{violation_count}] Frame {frame_count}: "
                      f"Véhicule dans la zone avec {persons_in_zone} piéton(s)!")
        else:
            # Pas de piéton = on update quand même le tracker mais sans violation
            _, vehicles_in_zone_bboxes = vehicle_tracker.update(
                vehicle_detections, crosswalk_polygon
            )
            # Reset le statut "violated" si le piéton est parti
            for vid in vehicle_tracker.vehicles:
                if vehicle_tracker.vehicles[vid].get("in_zone", False):
                    # Le véhicule est dans la zone mais pas de piéton = pas de violation
                    pass
        
        vehicles_in_zone = len(vehicles_in_zone_bboxes)
        current_violation = (persons_in_zone > 0 and vehicles_in_zone > 0)
        
        # ----- DESSIN DES DÉTECTIONS -----
        # Dessiner les piétons
        for bbox, label, color in person_bboxes:
            draw_detection(frame, bbox, label, color)
        
        # Dessiner les véhicules
        for bbox, class_name, conf in vehicle_bboxes:
            in_zone = bbox in vehicles_in_zone_bboxes or is_in_zone(bbox, crosswalk_polygon)
            
            if in_zone and persons_in_zone > 0:
                label = f"{class_name} {conf:.0%} [VIOLATION]"
                color = COLOR_RED
                is_violation = True
            elif in_zone:
                label = f"{class_name} {conf:.0%} [IN ZONE]"
                color = COLOR_ORANGE
                is_violation = False
            else:
                label = f"{class_name} {conf:.0%}"
                color = COLOR_BLUE
                is_violation = False
            
            draw_detection(frame, bbox, label, color, is_violation=is_violation)
        
        # ----- DESSIN SUR L'IMAGE -----
        frame = draw_crosswalk_polygon(frame, crosswalk_polygon, 
                                       violation=current_violation)
        
        frame = draw_status_panel(frame, persons_in_zone, vehicles_in_zone,
                                   current_violation, violation_count)
        
        if current_violation:
            frame = draw_violation_alert(frame)
        
        # ----- AFFICHAGE -----
        if args.show:
            cv2.imshow("Smart Crosswalk Monitor", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Arrêt demandé par l'utilisateur.")
                break
            elif key == ord('s'):
                screenshot_path = f"screenshots/screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot sauvegardé: {screenshot_path}")
            elif key == ord('p'):
                print("[INFO] Pause - Appuie sur n'importe quelle touche...")
                cv2.waitKey(0)
        
        # ----- SAUVEGARDE -----
        if writer is not None:
            writer.write(frame)
    
    # ================================
    # 7. Nettoyage
    # ================================
    print("=" * 60)
    print(f"[RÉSULTATS] Frames traitées: {frame_count}")
    print(f"[RÉSULTATS] Violations détectées: {violation_count}")
    
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Vidéo sauvegardée.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
