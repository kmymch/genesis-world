import os
import argparse
import pickle
import time
from pathlib import Path
import math

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

import genesis as gs
from env import GraspEnv
from eval import load_rl_policy
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode, InterpolationSpace

FOLLOWER_IP = "192.168.1.101"

def init_trossen_arm(gripper_mode=Mode.position):
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
        driver.set_gripper_mode(gripper_mode)
    except AttributeError:
        pass
    print("Trossen Arm connected and configured.")
    return driver

def angle_axis_to_quat_wxyz(angle_axis):
    """Convert angle-axis (rx, ry, rz) to quaternion (w, x, y, z)"""
    vec = np.array(angle_axis)
    angle = np.linalg.norm(vec)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = vec / angle
    q_xyz = np.sin(angle / 2) * axis
    q_w = np.cos(angle / 2)
    return np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]])

def get_ee_pose(driver):
    """Get current EE link pose in Genesis format [x, y, z, w, qx, qy, qz]"""
    cartesian = driver.get_cartesian_positions()
    pos = cartesian[:3]
    angle_axis = cartesian[3:]
    quat = angle_axis_to_quat_wxyz(angle_axis)
    ee_pose = np.concatenate([pos, quat])
    return torch.tensor(ee_pose, dtype=torch.float32, device=gs.device).unsqueeze(0)

def estimate_object_pose(color_image, depth_frame, intrinsics, T_cam_to_base, calib_camera_matrix=None, calib_dist_coeffs=None):
    """
    Estimates the 3D pose of the ArUco marker (ID=23) and transforms it to the robot base frame.
    If intrinsic calibration is provided, it uses it for accurate distortion removal.
    
    Returns: (3,) numpy array [x, y, z] in robot base frame
        obj_quat: (4,) numpy array [w, x, y, z] in robot base frame
        tvec_cam: (3,) numpy array [x, y, z] in camera frame
        rvec_cam: (3,) numpy array [x, y, z] in camera frame
    """
    # 1. Detect ArUco Marker (OpenCV 4.7.0+ API)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(color_image)
    
    if ids is not None and len(ids) > 0:
        # Find index of ID=23
        target_idx = -1
        for i in range(len(ids)):
            if ids[i][0] == 23:
                target_idx = i
                break
                
        if target_idx != -1:
            # The printed black area is 24mm (inside a 30mm total cut-out)
            marker_size = 0.024 
            
            if calib_camera_matrix is not None and calib_dist_coeffs is not None:
                camera_matrix = calib_camera_matrix
                dist_coeffs = calib_dist_coeffs
            else:
                camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                          [0, intrinsics.fy, intrinsics.ppy],
                                          [0, 0, 1]], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1)) # Assuming minimal distortion
    
            # OpenCV 4.7+ removed estimatePoseSingleMarkers, use solvePnP
            half_size = marker_size / 2.0
            obj_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)
    
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[target_idx][0], camera_matrix, dist_coeffs)
            
            tvec_cam = tvec.flatten()
            rvec_cam = rvec.flatten()
            
            # --- Use RealSense Depth for Robust Z Estimation ---
            c = corners[target_idx][0]
            u = int(np.mean(c[:, 0]))
            v = int(np.mean(c[:, 1]))
            
            if 0 <= u < color_image.shape[1] and 0 <= v < color_image.shape[0]:
                z_depth = depth_frame.get_distance(u, v)
                if z_depth > 0:
                    # ロボットアームがマーカーの上に被さると、Depthセンサがアームの距離を拾ってしまう
                    # solvePnPが計算したマーカーの距離(tvec_cam[2])と大きく乖離している場合はアームの誤検知とみなす
                    if abs(z_depth - tvec_cam[2]) < 0.05:
                        tvec_cam[2] = z_depth
                    # 乖離している(アームが被さっている)場合は、更新せずsolvePnPの深度(少しノイジーだがアームの影響を受けない)にフォールバック
            # ---------------------------------------------------
            
            
            # Draw for visualization
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec_cam, tvec_cam, 0.05)
            
            # Construct marker transform in camera frame
            T_marker_to_cam = np.eye(4)
            T_marker_to_cam[:3, :3] = R.from_rotvec(rvec_cam).as_matrix()
            T_marker_to_cam[:3, 3] = tvec_cam
            
            # Offset from marker (top face) to block center (15mm down along marker's Z axis)
            T_obj_to_marker = np.eye(4)
            T_obj_to_marker[2, 3] = -0.015 
            
            # Transform to base frame
            T_obj_to_base = T_cam_to_base @ T_marker_to_cam @ T_obj_to_marker
            
            obj_pos_base = T_obj_to_base[:3, 3]
            
            # Enforce upright orientation and match simulation 180deg X rotation
            euler_zyx = R.from_matrix(T_obj_to_base[:3, :3]).as_euler('ZYX')
            yaw = euler_zyx[0]
            r_yaw = R.from_euler('Z', yaw)
            r_down = R.from_euler('X', 180, degrees=True)
            r_target = r_yaw * r_down  # Apply r_down first, then r_yaw (Scipy composition order)
            quat_xyzw = r_target.as_quat()
            
            obj_quat_base = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]) # w, x, y, z
    
            cv2.imshow("ArUco Detection", color_image)
            cv2.waitKey(1)
            return obj_pos_base, obj_quat_base, tvec_cam, rvec_cam

    cv2.imshow("ArUco Detection", color_image)
    cv2.waitKey(1)
    
    return None, None, None, None

def run_extrinsic_calibration(driver, cap_top, align, intrinsics, calib_camera_matrix, calib_dist_coeffs):
    print("\n--- Starting Extrinsic Calibration ---")
    print("Moving to initial posture (all 0.0)...")
    driver.set_all_positions([0.0]*7, goal_time=3.0, blocking=True)
    time.sleep(1.0)
    
    print("Opening gripper (position control)...")
    driver.set_gripper_mode(Mode.position)
    driver.set_gripper_position(0.03)
    time.sleep(5.0)
    
    print("Closing gripper to grasp 3cm cube (external effort)...")
    driver.set_gripper_mode(Mode.external_effort)
    driver.set_gripper_external_effort(-20.0) # Positive effort to close and hold
    time.sleep(2.0)
    
    xs = [0.30, 0.45, 0.60]
    ys = [-0.20, 0.0, 0.20]
    zs = [0.20, 0.25, 0.30]
    
    base_target = [0.30, 0.0, 0.20, 0.0, 0.0, 0.0]
    print(f"Moving to base posture {base_target[:3]}...")
    driver.set_cartesian_positions(goal_positions=base_target, interpolation_space=InterpolationSpace.cartesian, goal_time=3.0, blocking=True)
    time.sleep(1.0)
    
    marker_cam_list = []
    marker_base_list = []
    T_cam_to_base_dummy = np.eye(4)
    
    # Generate snake-like path to avoid large diagonal jumps
    for i, x in enumerate(xs):
        current_ys = ys if i % 2 == 0 else ys[::-1]
        for j, y in enumerate(current_ys):
            current_zs = zs if j % 2 == 0 else zs[::-1]
            for z in current_zs:
                target_pos = [x, y, z, 0.0, 0.0, 0.0]
                print(f"Moving to {target_pos[:3]}...")
                driver.set_cartesian_positions(goal_positions=target_pos, interpolation_space=InterpolationSpace.cartesian, goal_time=1.5, blocking=True)
                time.sleep(1.0)
                
                frames = cap_top.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                    
                color_image = np.asanyarray(color_frame.get_data())
                _, _, tvec_cam, _ = estimate_object_pose(color_image, depth_frame, intrinsics, T_cam_to_base_dummy, calib_camera_matrix, calib_dist_coeffs)
                
                if tvec_cam is not None:
                    cartesian = driver.get_cartesian_positions()
                    tcp_pos = np.array(cartesian[:3])
                    
                    # Marker is 1.5cm above the center of the 3cm cube
                    marker_pos_base = tcp_pos + np.array([0.0, 0.0, 0.015])
                    
                    marker_cam_list.append(tvec_cam)
                    marker_base_list.append(marker_pos_base)
                    print(f"  Captured: TCP {tcp_pos.round(3)} -> Marker Base {marker_pos_base.round(3)} | Camera {tvec_cam.round(3)}")
                else:
                    print("  Marker not detected. Skipping.")
                    
    if len(marker_cam_list) < 3:
        print("Not enough points to calibrate (need at least 3). Calibration failed.")
    else:
        cam_pts = np.array(marker_cam_list)
        base_pts = np.array(marker_base_list)
        
        cam_center = np.mean(cam_pts, axis=0)
        base_center = np.mean(base_pts, axis=0)
        
        cam_centered = cam_pts - cam_center
        base_centered = base_pts - base_center
        
        rot, rmsd = R.align_vectors(base_centered, cam_centered)
        t = base_center - rot.apply(cam_center)
        
        T_cam_to_base = np.eye(4)
        T_cam_to_base[:3, :3] = rot.as_matrix()
        T_cam_to_base[:3, 3] = t
        
        print(f"\nCalibration complete. RMSD: {rmsd:.4f}")
        print("T_cam_to_base:\n", T_cam_to_base)
        
        calib_dir = os.path.join(os.path.dirname(__file__), "calibration")
        os.makedirs(calib_dir, exist_ok=True)
        save_path = os.path.join(calib_dir, "T_cam_to_base.npy")
        np.save(save_path, T_cam_to_base)
        print(f"Saved calibration matrix to {save_path}")
    
    print("Moving back to base posture...")
    driver.set_cartesian_positions(goal_positions=base_target, interpolation_space=InterpolationSpace.cartesian, goal_time=3.0, blocking=True)
    
    print("Opening gripper...")
    driver.set_gripper_mode(Mode.position)
    driver.set_gripper_position(0.03)
    time.sleep(1.0)
    print("Extrinsic Calibration Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default=Path(__file__).resolve().parent.name)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without sending to robot")
    parser.add_argument("--top-cam", type=str, default="243322072171", help="Serial number for top camera")
    parser.add_argument("--target-x", type=float, default=0.4, help="X coordinate of the target region (meters)")
    parser.add_argument("--target-y", type=float, default=0.0, help="Y coordinate of the target region (meters)")
    parser.add_argument("--calibrate-extrinsic", action="store_true", help="Run extrinsic camera calibration routine")
    args = parser.parse_args()

    # --- Setup Camera for both modes ---
    print(f"Opening Top Camera (SN: {args.top_cam}) for ArUco Tracking...")
    ctx = rs.context()
    cap_top = rs.pipeline(ctx)
    cfg_top = rs.config()
    cfg_top.enable_device(args.top_cam)
    cfg_top.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg_top.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = cap_top.start(cfg_top)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    print("Waiting for auto-exposure to settle...")
    for _ in range(30):
        cap_top.wait_for_frames()

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Load Camera Intrinsic Calibration if exists
    calib_dir = os.path.join(os.path.dirname(__file__), "calibration")
    camera_matrix_path = os.path.join(calib_dir, "camera_matrix.npy")
    dist_coeffs_path = os.path.join(calib_dir, "dist_coeffs.npy")
    
    if os.path.exists(camera_matrix_path) and os.path.exists(dist_coeffs_path):
        print(f"Loaded camera_matrix and dist_coeffs from {calib_dir}")
        calib_camera_matrix = np.load(camera_matrix_path)
        calib_dist_coeffs = np.load(dist_coeffs_path)
    else:
        print("No intrinsic calibration found. Using default RealSense intrinsics.")
        calib_camera_matrix = None
        calib_dist_coeffs = None

    if args.calibrate_extrinsic:
        if args.dry_run:
            print("Cannot run calibration in dry-run mode.")
            cap_top.stop()
            return
        driver = init_trossen_arm(gripper_mode=Mode.position)
        run_extrinsic_calibration(driver, cap_top, align, intrinsics, calib_camera_matrix, calib_dist_coeffs)
        cap_top.stop()
        return

    # === Normal Execution Mode ===
    gs.init()

    # Load RL config instead of BC
    log_dir = Path("logs") / f"{args.exp_name}_rl"

    print("Loading configurations...")
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)

    # Override for sim2real
    env_cfg["num_envs"] = 1
    env_cfg["visualize_camera"] = False
    env_cfg["show_visual_helpers"] = False

    print("Initializing dummy environment for policy...")
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=False,
    )

    print("Loading RL policy...")
    policy = load_rl_policy(env, rl_train_cfg, log_dir)

    if not args.dry_run:
        driver = init_trossen_arm(gripper_mode=Mode.position)
    else:
        print("[DRY-RUN] Trossen Arm driver not initialized.")
        driver = None

    ctrl_dt = env_cfg["ctrl_dt"]
    max_sim_step = int(env_cfg["episode_length_s"] / ctrl_dt)

    if driver:
        print("Moving to initial posture before starting the policy...")
        initial_positions = [0.0, math.pi/6, math.pi/6, 0.0, 0.0, 0.0, 0.0]
        driver.set_all_positions(initial_positions, goal_time=3.0, blocking=True)
        time.sleep(1.0)

    print("Starting Sim2Real RL control loop...")
    
    # Initialize previous state for marker tracking loss
    prev_obj_pos = np.array([0.4, 0.0, 0.015])
    prev_obj_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    filtered_obj_pos = None
    ema_alpha = 0.15  # Smoothing factor
    
    initial_obj_pose_found = False
    has_grasped = False
    
    # --- Load Extrinsic Calibration ---
    t_cam_to_base_path = os.path.join(calib_dir, "T_cam_to_base.npy")
    if os.path.exists(t_cam_to_base_path):
        print(f"Loaded T_cam_to_base from {t_cam_to_base_path}")
        T_cam_to_base = np.load(t_cam_to_base_path)
    else:
        print("No extrinsic calibration found. Using hardcoded defaults.")
        rot = R.from_euler('xyz', [180.0, 0.0, 90.0], degrees=True).as_matrix()
        T_cam_to_base = np.eye(4)
        T_cam_to_base[:3, :3] = rot
        T_cam_to_base[:3, 3] = [0.405, -0.005, 0.863]

    try:
        with torch.no_grad():
            for step in range(max_sim_step):
                start_time = time.time()
    
                # 1. Capture Camera Image & Estimate Pose
                frames = cap_top.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    # Estimate Object Pose using RealSense + solvePnP (with optional intrinsic calib)
                    est_pos, est_quat, _, _ = estimate_object_pose(color_image, depth_frame, intrinsics, T_cam_to_base, calib_camera_matrix, calib_dist_coeffs)
                    
                    if not initial_obj_pose_found and est_pos is not None:
                        # 最初の1フレームだけ座標を取得して固定する
                        obj_pos = est_pos
                        obj_pos[2] = 0.015 # Zはテーブルの高さに強制
                        obj_quat = est_quat
                        
                        prev_obj_pos = obj_pos.copy()
                        prev_obj_quat = obj_quat.copy()
                        initial_obj_pose_found = True
                        print(f"Initial Object Pose Locked: {obj_pos.round(3)}")
                    elif initial_obj_pose_found:
                        # 動作確認のため、いかなる状態でも最初に固定した座標をずっと使い続ける
                        obj_pos, obj_quat = prev_obj_pos, prev_obj_quat
                    else:
                        # まだ1フレームも取得できていない場合
                        obj_pos, obj_quat = prev_obj_pos, prev_obj_quat
                else:
                    obj_pos, obj_quat = prev_obj_pos, prev_obj_quat
    
                # 2. Get Robot State
                if driver:
                    cartesian = driver.get_cartesian_positions()
                    tcp_curr_pos = np.array(cartesian[:3])
                    curr_aa = np.array(cartesian[3:])
                    R_curr = R.from_rotvec(curr_aa)
                    
                    ee_curr_pos = tcp_curr_pos
                    
                    ee_quat_xyzw = R_curr.as_quat()
                    ee_quat_wxyz = np.array([ee_quat_xyzw[3], ee_quat_xyzw[0], ee_quat_xyzw[1], ee_quat_xyzw[2]])
                    ee_pose_full = torch.tensor(np.concatenate([ee_curr_pos, ee_quat_wxyz]), dtype=torch.float32, device=gs.device).unsqueeze(0)
                else:
                    ee_pose_full = torch.tensor([[0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]], device=gs.device)
    
                ee_pos = ee_pose_full[:, :3]
                ee_quat = ee_pose_full[:, 3:7]
    
                # 3. Construct 13D Observation Vector
                target_pos = torch.tensor([[args.target_x, args.target_y, 0.0005]], device=gs.device)
                
                obs_components = [
                    ee_pos,      # 3D
                    ee_quat,     # 4D
                    torch.tensor(obj_pos, dtype=torch.float32, device=gs.device).unsqueeze(0),
                    target_pos,
                ]
                obs_buf = torch.cat(obs_components, dim=-1) # Total 13D
                obs_dict = {"policy": obs_buf}
    
                # 4. Inference
                actions = policy(obs_dict)
                scaled_actions = actions * env.action_scales
                action = scaled_actions[0].cpu().numpy()
    
                delta_pos = action[:3]
                delta_rpy = action[3:6]
                gripper_action = action[6]
                
                if driver:
                    cartesian = driver.get_cartesian_positions()
                    curr_pos = np.array(cartesian[:3])
                    curr_aa = np.array(cartesian[3:])
                    # target_pos_cart is the target for TCP
                    target_tcp_pos = ee_curr_pos + delta_pos
    
                    # Rotational update (Intrinsic local frame delta)
                    r_curr = R.from_rotvec(curr_aa)
                    r_delta = R.from_euler('xyz', delta_rpy)
                    r_target = r_curr * r_delta
                    target_aa = r_target.as_rotvec()
    
                    target_cartesian = target_tcp_pos.tolist() + target_aa.tolist()
    
                    gripper_open = env.robot._gripper_open_dof
                    gripper_close = env.robot._gripper_close_dof
                    gripper_target = (gripper_action + 1.0) / 2.0 * (gripper_open - gripper_close) + gripper_close
    
                    if not args.dry_run:
                        driver.set_cartesian_positions(
                            goal_positions=target_cartesian,
                            interpolation_space=InterpolationSpace.cartesian,
                            goal_time=0.5,
                            blocking=True
                        )
                        try:
                            driver.set_gripper_position(gripper_target)
                        except AttributeError:
                            pass
                
                if step % 1 == 0 or args.dry_run:
                    ee_pose_np = ee_pose_full.cpu().numpy()[0]
                    target_pos_np = target_pos.cpu().numpy()[0]
                    raw_actions_np = actions.cpu().numpy()[0]
                    
                    print(f"\nStep {step}/{max_sim_step}")
                    print("  --- Observation (Input to Policy) ---")
                    print(f"    EE Pos      : {ee_pose_np[:3].round(4)}")
                    print(f"    EE Quat     : {ee_pose_np[3:7].round(4)}")
                    print(f"    Obj Pos     : {obj_pos.round(4)}")
                    print(f"    Target Pos  : {target_pos_np.round(4)}")
                    print("  --- Action (Output from Policy) ---")
                    print(f"    Raw Action  : {raw_actions_np.round(4)}")
                    print(f"    Scaled      : {action.round(4)}")
    
                elapsed = time.time() - start_time
                wait_time_ms = int((0.1 - elapsed) * 1000)
                if wait_time_ms > 0:
                    cv2.waitKey(wait_time_ms)
                else:
                    cv2.waitKey(1)
    
    except KeyboardInterrupt:
        print('\n[Ctrl+C] Interrupted by user.')
        if driver:
            print('Moving all joints to 0.0 before exiting...')
            driver.set_all_positions([0.0]*7, goal_time=3.0, blocking=True)
            time.sleep(1.0)
    finally:
            cap_top.stop()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
