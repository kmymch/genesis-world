import cv2
import numpy as np
from PIL import Image

def generate_aruco_pdf(output_path="aruco_markers_30mm_exact.pdf"):
    # 300 DPI calculations
    dpi = 300
    px_per_mm = dpi / 25.4
    
    # A4 size in mm
    a4_width_mm = 210
    a4_height_mm = 297
    
    # Canvas size in pixels
    width_px = int(a4_width_mm * px_per_mm)
    height_px = int(a4_height_mm * px_per_mm)
    
    # Create white A4 canvas
    canvas = np.ones((height_px, width_px), dtype=np.uint8) * 255
    
    # Size settings
    total_size_mm = 30
    marker_size_mm = 24  # Leaves 3mm white margin on all sides
    
    total_px = int(total_size_mm * px_per_mm)
    marker_px = int(marker_size_mm * px_per_mm)
    margin_inside_px = (total_px - marker_px) // 2
    
    # Grid Spacing
    margin_px = int(20 * px_per_mm) # 20mm outer margin
    spacing_px = int(10 * px_per_mm) # 10mm spacing between total blocks
    
    cols = (width_px - 2 * margin_px + spacing_px) // (total_px + spacing_px)
    rows = (height_px - 2 * margin_px + spacing_px) // (total_px + spacing_px)
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    marker_id = 0
    
    for i in range(rows):
        for j in range(cols):
            x = margin_px + j * (total_px + spacing_px)
            y = margin_px + i * (total_px + spacing_px)
            
            # Draw cut guide (light gray 30x30mm box)
            cv2.rectangle(canvas, (x, y), (x + total_px, y + total_px), (200,), 1)
            
            # Generate marker exactly at target pixel size (24mm)
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
            
            # Place marker inside the 30mm box with margin
            mx = x + margin_inside_px
            my = y + margin_inside_px
            canvas[my:my+marker_px, mx:mx+marker_px] = marker_img
            
            # Draw label (ID) above marker
            label = f"ID: {marker_id}"
            cv2.putText(canvas, label, (x, y - int(2 * px_per_mm)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,), 1, cv2.LINE_AA)
            
            marker_id += 1

    # Save using PIL to set exact DPI metadata in PDF
    pil_img = Image.fromarray(canvas)
    pil_img.save(output_path, "PDF", resolution=dpi)
    print(f"Saved exact size PDF to {output_path}")

if __name__ == "__main__":
    generate_aruco_pdf()
