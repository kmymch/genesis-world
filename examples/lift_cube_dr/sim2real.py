import argparse
import pickle
import time
from pathlib import Path
import math

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import genesis as gs
from env import GraspEnv
from eval import load_bc_policy
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode, InterpolationSpace

FOLLOWER_IP = "192.168.1.101"

def init_trossen_arm():
    driver = TrossenArmDriver()
    driver.set_manual_ip(FOLLOWER_IP)
    print(f"Connecting to Trossen Arm at {FOLLOWER_IP}...")
    driver.configure(
        model=Model.wxai_v0,
        end_effector=StandardEndEffector.wxai_v0_follower,
        serv_ip=FOLLOWER_IP,
        clear_error=True
    )
    driver.set_arm_modes(Mode.position)
    # Gripper typically uses position mode as well
    try:
        driver.set_gripper_mode(Mode.position)
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
    """Get current EE pose in Genesis format [x, y, z, w, qx, qy, qz]"""
    cartesian = driver.get_cartesian_positions()
    pos = cartesian[:3]
    angle_axis = cartesian[3:]
    quat = angle_axis_to_quat_wxyz(angle_axis)
    ee_pose = np.concatenate([pos, quat])
    return torch.tensor(ee_pose, dtype=torch.float32, device=gs.device).unsqueeze(0)

def capture_image(cap, res=(64, 48)):
    """Capture and format image for BC policy"""
    ret, frame = cap.read()
    if not ret:
        return np.zeros((3, res[1], res[0]), dtype=np.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, res)
    frame = np.transpose(frame, (2, 0, 1))
    return frame.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default=Path(__file__).resolve().parent.name)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without sending to robot")
    parser.add_argument("--top-cam", type=int, default=0, help="Camera ID for top camera")
    parser.add_argument("--wrist-cam", type=int, default=1, help="Camera ID for wrist camera")
    args = parser.parse_args()

    gs.init()

    log_dir = Path("logs") / f"{args.exp_name}_bc"

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

    print("Loading BC policy...")
    policy = load_bc_policy(env, bc_train_cfg, log_dir)
    policy.eval()

    if not args.dry_run:
        driver = init_trossen_arm()
    else:
        print("[DRY-RUN] Trossen Arm driver not initialized.")
        driver = None

    print(f"Opening cameras (Top: {args.top_cam}, Wrist: {args.wrist_cam})...")
    cap_top = cv2.VideoCapture(args.top_cam)
    cap_wrist = cv2.VideoCapture(args.wrist_cam)

    ctrl_dt = env_cfg["ctrl_dt"]
    max_sim_step = int(env_cfg["episode_length_s"] / ctrl_dt)

    print("Starting Sim2Real control loop...")
    with torch.no_grad():
        for step in range(max_sim_step):
            start_time = time.time()

            # 1. Get Observations
            top_img = capture_image(cap_top, res=env_cfg["top_cam_resolution"])
            wrist_img = capture_image(cap_wrist, res=env_cfg["wrist_cam_resolution"])
            
            top_obs = torch.tensor(top_img, device=gs.device).unsqueeze(0)
            wrist_obs = torch.tensor(wrist_img, device=gs.device).unsqueeze(0)
            
            if driver:
                ee_pose = get_ee_pose(driver)
            else:
                # Mock EE pose for dry-run if no driver
                ee_pose = torch.tensor([[0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]], device=gs.device)

            # 2. Inference
            actions = policy(top_obs, wrist_obs, ee_pose)
            # Genesis outputs actions scaled by action_scales, wait, env.step() multiplies by action_scales.
            # In eval.py, env.step(actions) is called. In env.py:
            # def step(self, actions: torch.Tensor):
            #     actions = self.rescale_action(actions)
            # So the raw output from policy is UN-scaled action in [-1, 1] range!
            # We must rescale it before applying.
            scaled_actions = actions * env.action_scales
            action = scaled_actions[0].cpu().numpy()

            delta_pos = action[:3]
            delta_rpy = action[3:6]
            gripper_action = action[6] # raw unscaled or scaled? 
            # In env.py:
            # gripper_action = action[:, 6]
            # gripper_target = (gripper_action + 1.0) / 2.0 * (open - close) + close
            # Wait, is gripper dimension scaled by action_scales?
            # Usually action_scales is [scale_p, scale_p, scale_p, scale_r, scale_r, scale_r, scale_g]
            
            if driver:
                cartesian = driver.get_cartesian_positions()
                curr_pos = np.array(cartesian[:3])
                curr_aa = np.array(cartesian[3:])
                
                target_pos = curr_pos + delta_pos

                # Rotational update
                # Genesis uses Hamilton product. For now, let's use scipy Rotation.
                r_curr = R.from_rotvec(curr_aa)
                r_rel = R.from_euler('xyz', delta_rpy) # xyz roll pitch yaw
                r_target = r_curr * r_rel # Apply relative rotation
                target_aa = r_target.as_rotvec()

                target_cartesian = target_pos.tolist() + target_aa.tolist()

                # Gripper
                gripper_open = env.robot._gripper_open_dof
                gripper_close = env.robot._gripper_close_dof
                # In env.py apply_action(): gripper_target = (gripper_action + 1.0) / 2.0 * (self._gripper_open_dof - self._gripper_close_dof) + self._gripper_close_dof
                # Here `gripper_action` is the SCALED action. Wait, does genesis scale the gripper action?
                # Usually `env_cfg["action_scales"]` is 1.0 for all, but let's check `cfgs.pkl`.
                # We will just use the exact logic from env.py.
                
                # Unscaled action for gripper? In apply_action it uses the action directly passed (which is scaled).
                gripper_target = (gripper_action + 1.0) / 2.0 * (gripper_open - gripper_close) + gripper_close

                if not args.dry_run:
                    driver.set_cartesian_positions(
                        goal_positions=target_cartesian,
                        interpolation_space=InterpolationSpace.cartesian,
                        goal_time=ctrl_dt,
                        blocking=False
                    )
                    # Note: set_gripper_position might be different
                    try:
                        driver.set_gripper_position(gripper_target)
                    except AttributeError:
                        pass
            
            # Print state for debugging
            if step % 10 == 0 or args.dry_run:
                print(f"Step {step}/{max_sim_step}")
                print(f"  Actions (raw)   : {actions[0].cpu().numpy().round(3)}")
                print(f"  Actions (scaled): {action.round(3)}")
                if driver:
                    print(f"  Target Cart     : {[round(v, 3) for v in target_cartesian]}")
                    print(f"  Target Gripper  : {round(gripper_target, 3)}")

            # Wait to match control frequency
            elapsed = time.time() - start_time
            if elapsed < ctrl_dt:
                time.sleep(ctrl_dt - elapsed)

    cap_top.release()
    cap_wrist.release()
    print("Done.")

if __name__ == "__main__":
    main()
