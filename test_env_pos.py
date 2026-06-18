import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene()
plane = scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(pos=(0,0,0), size=(0.1, 0.1, 0.1)))
scene.build(n_envs=10, env_spacing=(1.0, 1.0))
print("10 envs: box pos =\n", box.get_pos())
