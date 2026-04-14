"""
Outil de calibration : clique 4 points pour définir
une zone de passage piéton (polygone à 4 côtés).
Sauvegarde automatiquement dans config.py.
"""

import cv2
import sys
import numpy as np
import re


# Variables globales
points_clicked = []
frame_display = None
original_frame = None
frame_width = 0
frame_height = 0


def save_to_config(points_percent):
    """
    Sauvegarde les points dans config.py en remplaçant la ligne existante.
    """
    config_file = "config.py"
    
    # Lire le fichier config.py
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[ERREUR] Fichier {config_file} introuvable!")
        return False
    
    # Construire la nouvelle valeur
    new_value = "CROSSWALK_POLYGON_PERCENT = [\n"
    for i, (px, py) in enumerate(points_percent):
        comma = "," if i < 3 else ""
        new_value += f"    ({px}, {py}){comma}   # Point {i+1}\n"
    new_value += "]"
    
    # Pattern pour trouver et remplacer CROSSWALK_POLYGON_PERCENT
    pattern = r"CROSSWALK_POLYGON_PERCENT\s*=\s*\[.*?\]"
    
    if re.search(pattern, content, re.DOTALL):
        # Remplacer l'existant
        new_content = re.sub(pattern, new_value, content, flags=re.DOTALL)
    else:
        # Ajouter à la fin si n'existe pas
        new_content = content + "\n\n" + new_value + "\n"
    
    # Écrire le fichier
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"\n[OK] Configuration sauvegardée dans {config_file}!")
    return True


def mouse_callback(event, x, y, flags, param):
    global points_clicked, frame_display, original_frame
    global frame_width, frame_height
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_clicked) < 4:
            points_clicked.append((x, y))
            frame_display = original_frame.copy()
            
            # Dessiner les points déjà cliqués
            for i, (px, py) in enumerate(points_clicked, 1):
                cv2.circle(frame_display, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(frame_display, str(i), (px + 10, py + 10), 
                            0, 0.7, (255, 255, 255), 2)
            
            # Relier les points au fur et à mesure
            if len(points_clicked) > 1:
                for i in range(len(points_clicked) - 1):
                    cv2.line(frame_display, points_clicked[i], 
                            points_clicked[i + 1], (0, 255, 0), 2)
            
            # Si on a les 4 points, fermer le polygone et sauvegarder
            if len(points_clicked) == 4:
                cv2.line(frame_display, points_clicked[3], 
                        points_clicked[0], (0, 255, 0), 2)
                
                # Remplir le polygone en semi-transparent
                overlay = frame_display.copy()
                cv2.fillPoly(overlay, [np.array(points_clicked)], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame_display, 0.7, 0, frame_display)
                
                # Calculer les pourcentages pour chaque point
                points_percent = []
                for (px, py) in points_clicked:
                    px_pct = round(px / frame_width * 100)
                    py_pct = round(py / frame_height * 100)
                    points_percent.append((px_pct, py_pct))
                
                print(f"\n{'='*60}")
                print("Zone définie (4 points) :")
                for i, (pt, pct) in enumerate(zip(points_clicked, points_percent), 1):
                    print(f"  Point {i} : pixel {pt} -> {pct}%")
                print(f"{'='*60}")
                
                # Sauvegarder automatiquement
                if save_to_config(points_percent):
                    print("\nTu peux maintenant lancer:")
                    print("  python main.py --source <ta_video>")
                    print("\nAppuie 'Q' pour quitter ou 'R' pour recalibrer.")
                
            else:
                print(f"Point {len(points_clicked)}/4 ajouté. "
                      f"Clique encore {4 - len(points_clicked)} point(s).")


def main():
    global frame_display, original_frame, points_clicked
    global frame_width, frame_height
    
    if len(sys.argv) < 2:
        print("Usage: python calibrate_zone.py <chemin_video>")
        print("Exemple: python calibrate_zone.py videos/test.mp4")
        return
    
    source = sys.argv[1]
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Erreur: impossible de lire la vidéo '{source}'")
        return
    
    original_frame = frame.copy()
    frame_display = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    window_name = "Calibration - Clique 4 points (Q=quitter, R=reset)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("="*60)
    print("CALIBRATION DU PASSAGE PIÉTON")
    print("="*60)
    print("Instructions:")
    print("  1. Clique les 4 coins du passage piéton dans l'ordre")
    print("     (sens horaire ou anti-horaire)")
    print("  2. Les points seront reliés : 1->2->3->4->1")
    print("  3. La config sera sauvegardée automatiquement!")
    print("  4. Appuie 'R' pour recommencer, 'Q' pour quitter")
    print("="*60)
    
    while True:
        cv2.imshow(window_name, frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points_clicked = []
            frame_display = original_frame.copy()
            print("\n[RESET] Calibration réinitialisée. Clique 4 nouveaux points.")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
