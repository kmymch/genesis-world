import cv2
import numpy as np
import pyrealsense2 as rs
import os

def main():
    print("=== Homography Calibration ===")
    print("This script computes a 2D mapping from camera pixels to robot coordinates.")
    print("Please print the ChArUco board and place it flat on the table.")
    print("CRITICAL: Elevate the board so it sits exactly at the height of the cube's marker (3cm).")
    print("ALIGNMENT INSTRUCTIONS:")
    print("1. 印刷したA4用紙を「縦向き（Portrait）」にしてロボットの正面に置いてください。")
    print("2. 紙の下端（短い辺）がロボット側に向くようにします。")
    print("3. 紙の左下の角（マージンを含まない、一番左下の黒いマスの外側の角）のロボット座標(X, Y)を測ります。")
    
    try:
        base_x = float(input("紙の【左下角】のロボットX座標を入力してください（例: 0.3）: "))
        base_y = float(input("紙の【左下角】のロボットY座標を入力してください（例: 0.15）: "))
    except ValueError:
        print("Invalid input. Defaulting to X=0.3, Y=-0.1")
        base_x = 0.3
        base_y = -0.1

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
    
    print("\nCamera started. Press 'c' to capture and calibrate. Press 'q' to quit.")
    
    homography = None

    try:
        while True:
            frames = cap.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            display_frame = frame.copy()
            
            # Detect ChArUco board using new OpenCV 4.7+ API
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
            
            retval = 0
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)
                if charuco_ids is not None and len(charuco_ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids, (0, 255, 0))
                    retval = len(charuco_ids)
                    cv2.putText(display_frame, f"Found {retval} corners. Press 'c' to calibrate.", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Calibration", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                if charuco_ids is None or len(charuco_ids) < 4:
                    print("Not enough corners detected. Need at least 4.")
                    continue
                
                print(f"Captured! Computing homography using {retval} corners...")
                
                # Board's local 3D points
                board_pts_3d = board.getChessboardCorners()
                
                src_pts = []
                dst_pts = []
                
                for i in range(retval):
                    # Pixel coordinate
                    pixel_x, pixel_y = charuco_corners[i][0]
                    src_pts.append([pixel_x, pixel_y])
                    
                    # Physical local coordinate in board
                    corner_id = charuco_ids[i][0]
                    local_pt = board_pts_3d[corner_id]
                    
                    # A4用紙を縦置き（Portrait）で、下端をロボットに向けた場合：
                    # - ボードのY軸（紙の上方向） = ロボットのX軸（前方）
                    # - ボードのX軸（紙の右方向） = ロボットの-Y軸（右方）
                    robot_x = base_x + local_pt[1]
                    robot_y = base_y - local_pt[0]
                    
                    dst_pts.append([robot_x, robot_y])
                
                src_pts = np.array(src_pts, dtype=np.float32)
                dst_pts = np.array(dst_pts, dtype=np.float32)
                
                H, status = cv2.findHomography(src_pts, dst_pts)
                print("Homography Matrix:")
                print(H)
                
                out_path = os.path.join(os.path.dirname(__file__), "homography.npy")
                np.save(out_path, H)
                print(f"Saved to {out_path}")
                homography = H
                break

    finally:
        cap.stop()
        cv2.destroyAllWindows()
        
    if homography is not None:
        print("\nCalibration successful! You can now run sim2real.py")

if __name__ == "__main__":
    main()
