import time
import numpy as np
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

FOLLOWER_IP = "192.168.1.101"

def init_trossen_arm():
    driver = TrossenArmDriver()
    print(f"Connecting to Trossen Arm at {FOLLOWER_IP}...")
    driver.configure(
        model=Model.wxai_v0,
        end_effector=StandardEndEffector.wxai_v0_follower,
        serv_ip=FOLLOWER_IP,
        clear_error=True
    )
    driver.set_arm_modes(Mode.position)
    try:
        driver.set_gripper_mode(Mode.position)
    except AttributeError:
        pass
    print("Trossen Arm connected and configured.")
    return driver

def main():
    driver = init_trossen_arm()
    
    print("Moving all joints to 0.0 (initial posture)...")
    driver.set_all_positions([0.0] * 7, goal_time=3.0, blocking=True)
    time.sleep(1.0)
    
    print("\n--- Starting EE Pose Monitor (Ctrl+C to stop) ---")
    
    try:
        while True:
            # Get cartesian position from driver
            cartesian = driver.get_cartesian_positions()
            pos = cartesian[:3]
            angle_axis = cartesian[3:]
            
            print(f"EE Position (X, Y, Z): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"EE Rotation (Angle-Axis): [{angle_axis[0]:.4f}, {angle_axis[1]:.4f}, {angle_axis[2]:.4f}]")
            print("-" * 50)
            
            time.sleep(0.5) # 0.5秒おきに出力
            
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted by user.")
    finally:
        print("Done.")

if __name__ == "__main__":
    main()
