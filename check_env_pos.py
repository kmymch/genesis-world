import genesis as gs
import torch

gs.init(backend=gs.cpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(),
    show_viewer=False
)
plane = scene.add_entity(gs.morphs.Plane())
scene.build(n_envs=2048, env_spacing=(1.0, 1.0))

print("Env offsets:")
print(scene.env_offsets[:10])
print(scene.env_offsets[1020:1030])
print(scene.env_offsets[-10:])
