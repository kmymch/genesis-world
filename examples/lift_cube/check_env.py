import genesis as gs
import pickle
from pathlib import Path
from env import GraspEnv

def main():
    gs.init(backend=gs.cpu)
    log_dir = Path("logs/lift_cube_dr_bc")
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)
    
    env_cfg["num_envs"] = 1
    env = GraspEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=False)
    
    env.reset()
    try:
        qpos = env.robot.get_qpos()
    except Exception:
        qpos = env.robot._robot_entity.get_qpos()
    
    print("Initial qpos:", qpos)

if __name__ == "__main__":
    main()
