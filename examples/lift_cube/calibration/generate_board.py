import cv2
import numpy as np
import os
try:
    from PIL import Image
except ImportError:
    Image = None

def main():
    # Board configurations
    squares_x = 5  # changed to 5x7 so it fits on A4 with margins
    squares_y = 7
    square_length = 0.030  # 30 mm
    marker_length = 0.022  # 22 mm
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    
    # 300 DPI Settings
    DPI = 300
    pixels_per_mm = DPI / 25.4
    
    # Calculate exact pixel size of the board itself (no margins)
    board_w_mm = squares_x * (square_length * 1000)
    board_h_mm = squares_y * (square_length * 1000)
    
    board_w_px = int(np.round(board_w_mm * pixels_per_mm))
    board_h_px = int(np.round(board_h_mm * pixels_per_mm))
    
    # Generate the board image with exact physical dimensions and 0 margin
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0)
    
    # Create A4 canvas (210mm x 297mm) at 300 DPI
    a4_w_px = int(np.round(210 * pixels_per_mm))
    a4_h_px = int(np.round(297 * pixels_per_mm))
    a4_canvas = np.ones((a4_h_px, a4_w_px), dtype=np.uint8) * 255
    
    # Paste the board into the center of the A4 canvas
    start_x = (a4_w_px - board_w_px) // 2
    start_y = (a4_h_px - board_h_px) // 2
    a4_canvas[start_y:start_y+board_h_px, start_x:start_x+board_w_px] = board_img
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "charuco_board.png")
    pdf_path = os.path.join(out_dir, "charuco_board.pdf")
    
    # Save as PNG
    cv2.imwrite(png_path, a4_canvas)
    print(f"Saved ChArUco board PNG to {png_path}")
    print(f"Board size: {squares_x}x{squares_y} squares, Square Size: {square_length*1000}mm, Marker Size: {marker_length*1000}mm")
    
    # Save as PDF if PIL is available
    if Image is not None:
        im = Image.fromarray(a4_canvas)
        im.save(pdf_path, "PDF", resolution=DPI)
        print(f"Saved EXACT A4 ChArUco board PDF to {pdf_path}")
        print("IMPORTANT: Print this PDF at exactly 100% (Actual Size) on A4 paper.")

if __name__ == "__main__":
    main()
