import cv2
import sys
import numpy as np
import re

points_clicked = []
frame_display = None
original_frame = None
frame_width = 0
frame_height = 0

def save_to_config(points_percent):
    """Saves the calibrated crosswalk polygon points as percentage coordinates in the config.py file. This function reads the existing config.py file, updates or adds the CROSSWALK_POLYGON_PERCENT variable with the new points, and writes the changes back to the file. The points are saved in a format that can be easily imported and used by the main application for detection and processing."""
    config_file = "config.py"
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[ERROR] File {config_file} not found!")
        return False
    
    new_value = "CROSSWALK_POLYGON_PERCENT = [\n"
    for i, (px, py) in enumerate(points_percent):
        comma = "," if i < 3 else ""
        new_value += f"    ({px}, {py}){comma}   # Point {i+1}\n"
    new_value += "]"
    
    pattern = r"CROSSWALK_POLYGON_PERCENT\s*=\s*\[.*?\]"
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, new_value, content, flags=re.DOTALL)
    else:
        new_content = content + "\n\n" + new_value + "\n"
        
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"\n[OK] Configuration saved in {config_file}!")
    return True

def mouse_callback(event, x, y, flags, param):
    """Handles mouse click events for calibrating the crosswalk zone. When the user clicks on the video frame, this function records the click coordinates as points for defining the crosswalk polygon. It visually marks the clicked points and draws lines between them to form a polygon. Once 4 points are defined, it converts them to percentage coordinates, saves them to the config file, and provides feedback to the user."""
    global points_clicked, frame_display, original_frame, frame_width, frame_height
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_clicked) < 4:
            points_clicked.append((x, y))
            frame_display = original_frame.copy()
            
            for i, (px, py) in enumerate(points_clicked, 1):
                cv2.circle(frame_display, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(frame_display, str(i), (px + 10, py + 10), 0, 0.7, (255, 255, 255), 2)
            
            if len(points_clicked) > 1:
                for i in range(len(points_clicked) - 1):
                    cv2.line(frame_display, points_clicked[i], points_clicked[i + 1], (0, 255, 0), 2)
            
            if len(points_clicked) == 4:
                cv2.line(frame_display, points_clicked[3], points_clicked[0], (0, 255, 0), 2)
                overlay = frame_display.copy()
                cv2.fillPoly(overlay, [np.array(points_clicked)], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame_display, 0.7, 0, frame_display)
                
                points_percent = []
                for (px, py) in points_clicked:
                    px_pct = round(px / frame_width * 100)
                    py_pct = round(py / frame_height * 100)
                    points_percent.append((px_pct, py_pct))
                
                if save_to_config(points_percent):
                    print("\nPress 'Q' to quit or 'R' to recalibrate.")
            else:
                print(f"Point {len(points_clicked)}/4 ajouté.")

def main():
    """Main function to run the calibration tool. This function initializes the video capture from the provided source, sets up the display window and mouse callback for user interaction, and handles the main loop for displaying the video frame and processing user input. The user can click on the video frame to define the crosswalk polygon, and once 4 points are defined, they can save the configuration or reset it as needed."""
    global frame_display, original_frame, points_clicked, frame_width, frame_height
    
    if len(sys.argv) < 2:
        print("Usage: python calibrate_zone.py <video_path>")
        return
        
    source = sys.argv[1]
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return
    
    original_frame = frame.copy()
    frame_display = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    window_name = "Calibration (Q=quitter, R=reset)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        cv2.imshow(window_name, frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            points_clicked = []
            frame_display = original_frame.copy()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
