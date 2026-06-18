import genesis as gs
import torch

gs.init(backend=gs.cpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(),
    show_viewer=False
)
plane = scene.add_entity(gs.morphs.Plane())
morph = gs.morphs.Box(size=[0.1, 0.1, 0.1])
obj = scene.add_entity(morph)
scene.build(n_envs=2048, env_spacing=(1.0, 1.0))

print("Env 0 pos:", obj.get_pos()[0])
print("Env 9 pos:", obj.get_pos()[9])
print("Env 1024 pos:", obj.get_pos()[1024])
print("Env 2047 pos:", obj.get_pos()[2047])
