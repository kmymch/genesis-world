import genesis as gs
import torch

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
try:
    print("draw_debug_spheres exists:", hasattr(scene, "draw_debug_spheres"))
    import inspect
    print("signature:", inspect.signature(scene.draw_debug_spheres))
except Exception as e:
    print(e)
