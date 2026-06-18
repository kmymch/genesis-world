import genesis as gs

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
morph = gs.morphs.MJCF(
    file="/home/kmymch/ws/trossen_arm_mujoco/trossen_arm_mujoco/assets/wxai/wxai_follower.xml",
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
)
robot = scene.add_entity(morph=morph)
scene.build()

try:
    link = robot.get_link("ee_site")
    print("Success get_link('ee_site')")
except Exception as e:
    print("get_link error:", e)

try:
    print("Available links:", [l.name for l in robot.links])
except Exception as e:
    print("Failed to print links:", e)
