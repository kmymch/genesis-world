import cv2
import numpy as np
import pyrealsense2 as rs
import os

def main():
    print("=== Camera Intrinsic Calibration ===")
    print("This script computes the camera matrix and lens distortion coefficients.")
    print("Please hold the printed ChArUco board in front of the camera.")
    print("Tilt it at various angles and distances.")
    
    # Board configurations (MUST match generate_board.py)
    squares_x = 5
    squares_y = 7
    square_length = 0.030  # 30 mm
    marker_length = 0.022  # 22 mm
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    # Init RealSense
    ctx = rs.context()
    cap = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cap.start(cfg)
    
    all_obj_pts = []
    all_img_pts = []
    
    print("\nCamera started. Press 'c' to capture a frame. Press 'q' to finish and compute.")
    print("Goal: Capture 15-20 frames with the board at different angles/distances.")
    
    try:
        while True:
            frames = cap.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            display_frame = frame.copy()
            image_size = (frame.shape[1], frame.shape[0])
            
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
            
            retval = 0
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)
                if charuco_ids is not None and len(charuco_ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids, (0, 255, 0))
                    retval = len(charuco_ids)
            
            cv2.putText(display_frame, f"Captured: {len(all_img_pts)} frames", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Current corners: {retval}. Press 'c' to capture.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Intrinsic Calibration", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                if charuco_ids is None or len(charuco_ids) < 4:
                    print("Not enough corners detected. Move the board slower or closer.")
                    continue
                
                # Extract 3D points for detected corners
                board_pts = board.getChessboardCorners()
                objp = board_pts[charuco_ids.flatten()]
                
                all_obj_pts.append(objp)
                all_img_pts.append(charuco_corners)
                print(f"Frame {len(all_img_pts)} captured! ({len(charuco_ids)} corners)")

    finally:
        cap.stop()
        cv2.destroyAllWindows()
        
    if len(all_img_pts) > 0:
        print(f"\nComputing calibration using {len(all_img_pts)} frames...")
        
        # OpenCV 4.7+ API for calibration using standard calibrateCamera
        camera_matrix = np.eye(3, dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_pts, all_img_pts, image_size, camera_matrix, dist_coeffs
        )
        
        print(f"RMS Reprojection Error: {ret:.4f} pixels")
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)
        
        out_dir = os.path.dirname(__file__)
        np.save(os.path.join(out_dir, "camera_matrix.npy"), camera_matrix)
        np.save(os.path.join(out_dir, "dist_coeffs.npy"), dist_coeffs)
        print("\nSaved camera_matrix.npy and dist_coeffs.npy!")
        print("You can now run sim2real.py")
    else:
        print("No frames captured. Calibration aborted.")

if __name__ == "__main__":
    main()
