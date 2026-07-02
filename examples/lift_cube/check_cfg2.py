import pickle
from pathlib import Path

def main():
    log_dir = Path("logs/lift_cube_dr_bc")
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)
    print("env_cfg keys:", env_cfg.keys())
    for k, v in env_cfg.items():
        if "dof" in k or "joint" in k or "pos" in k:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
