import argparse
import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-cam", type=str, default="243322072171", help="Serial number for top camera")
    args = parser.parse_args()

    print(f"Opening Top Camera (SN: {args.top_cam}) for ArUco Tracking Test...")
    ctx = rs.context()
    cap_top = rs.pipeline(ctx)
    cfg_top = rs.config()
    cfg_top.enable_device(args.top_cam)
    cfg_top.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        profile = cap_top.start(cfg_top)
    except Exception as e:
        print(f"Error starting camera: {e}")
        print("Please check if the camera is connected and the serial number is correct.")
        return

    print("Waiting for auto-exposure to settle...")
    for _ in range(30):
        cap_top.wait_for_frames()

    # Setup ArUco detector (OpenCV 4.7.0+ API)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    print("Press 'q' to quit or 's' to save a screenshot.")

    try:
        while True:
            frames = cap_top.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Detect markers
            corners, ids, rejected = detector.detectMarkers(color_image)
            
            # Filter IDs to only show those <= 30
            if ids is not None:
                valid_corners = []
                valid_ids = []
                for i in range(len(ids)):
                    if ids[i][0] <= 30:
                        valid_corners.append(corners[i])
                        valid_ids.append(ids[i])
                
                if len(valid_ids) > 0:
                    valid_corners = tuple(valid_corners)
                    valid_ids = np.array(valid_ids)
                    cv2.aruco.drawDetectedMarkers(color_image, valid_corners, valid_ids)
                    
                    # Optional: Print out the detected IDs in the console
                    print(f"\rDetected IDs: {valid_ids.flatten().tolist()}", end="    ")
                else:
                    print("\rDetected IDs: []", end="        ")
            else:
                print("\rDetected IDs: []", end="        ")

            cv2.imshow("ArUco Test (Top Camera)", color_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("aruco_test_screenshot.jpg", color_image)
                print("\nScreenshot saved to aruco_test_screenshot.jpg")

    finally:
        cap_top.stop()
        cv2.destroyAllWindows()
        print("\nDone.")

if __name__ == "__main__":
    main()
