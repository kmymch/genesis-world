import genesis as gs
import pickle
from pathlib import Path
from env import GraspEnv
from PIL import Image
import numpy as np

def main():
    gs.init(backend=gs.cpu)
    log_dir = Path("logs/lift_cube_dr_bc")
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)
    
    env_cfg["num_envs"] = 1
    env = GraspEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=False)
    
    env.reset()
    
    # Render top camera
    top_cam = env.cam_top
    top_cam.render()
    img_rgb = top_cam.get_rgb()
    # image is returned as a tensor or numpy array, shape might be (H,W,3)
    if hasattr(img_rgb, 'cpu'):
        img_rgb = img_rgb.cpu().numpy()
    if img_rgb.dtype == np.float32:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    
    im = Image.fromarray(img_rgb)
    im.save("/home/kmymch/ws/genesis-world/examples/lift_cube_dr/sim_top_cam_0.png")
    print("Saved simulation top camera image to sim_top_cam_0.png")

if __name__ == "__main__":
    main()
