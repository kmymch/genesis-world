import genesis as gs
gs.init()
scene = gs.Scene(show_viewer=False)
scene.build(n_envs=10, env_spacing=(3.0, 3.0))
print("env_origins shape:", scene.env_origins.shape)
print("env_0 origin:", scene.env_origins[0])
