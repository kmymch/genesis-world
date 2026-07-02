import trossen_arm
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

FOLLOWER_IP = "192.168.1.101"

def main():
    driver = TrossenArmDriver()
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
    
    # Move all joints to 0
    print("Moving all joints to 0.0...")
    driver.set_all_positions([0.0] * 7, goal_time=3.0, blocking=True)
    
    cartesian = driver.get_cartesian_positions()
    print("Cartesian array length:", len(cartesian))
    print("Cartesian array:", cartesian)
    
    # Try to see what it is
    print("x, y, z:", cartesian[:3])
    print("rot part:", cartesian[3:])

if __name__ == "__main__":
    main()
