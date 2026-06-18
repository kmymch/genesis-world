import genesis as gs
gs.init()
scene = gs.Scene()
obj = scene.add_entity(gs.morphs.Box(pos=(0, 0, 0), size=(0.1, 0.1, 0.1), fixed=True))
scene.build()
methods = [m for m in dir(obj) if not m.startswith('_')]
print("Entity methods:", methods)
