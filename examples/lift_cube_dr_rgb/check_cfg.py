import pickle
from pathlib import Path

def main():
    log_dir = Path("logs/lift_cube_dr_bc")
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)
    print("robot_cfg init_qpos:", robot_cfg.get("init_qpos"))

if __name__ == "__main__":
    main()
