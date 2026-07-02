import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import time

def capture_images():
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    print(f"Found {len(devices)} RealSense devices.")

    pipelines = []
    
    # Create a pipeline for each device
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        print(f"Initializing device: {name} (S/N: {serial})")
        
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        
        # 640x480 (アスペクト比 4:3) で取得
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append((serial, name, pipeline))

    # カメラの自動露出が安定するまで少し待つ (数フレーム空読みする)
    print("Waiting for auto-exposure to settle...")
    for _ in range(30):
        for _, _, pipeline in pipelines:
            pipeline.wait_for_frames()

    # 保存先ディレクトリの作成
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(current_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    # 各カメラから画像を1枚取得して保存
    print("Capturing images...")
    for serial, name, pipeline in pipelines:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print(f"Failed to capture color frame from device {serial}")
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        
        formatted_name = name.replace(" ", "_")
        
        # 元サイズの保存
        filename = os.path.join(out_dir, f"{formatted_name}_{serial}.png")
        cv2.imwrite(filename, color_image)
        print(f"Saved {filename}")

        # 64x48にリサイズして保存 (アスペクト比4:3維持)
        color_image_resized = cv2.resize(color_image, (64, 48))
        filename_resized = os.path.join(out_dir, f"{formatted_name}_{serial}_64x48.png")
        cv2.imwrite(filename_resized, color_image_resized)
        print(f"Saved {filename_resized}")

    # パイプラインの停止
    for _, _, pipeline in pipelines:
        pipeline.stop()
        
    print("Done.")

if __name__ == "__main__":
    capture_images()
