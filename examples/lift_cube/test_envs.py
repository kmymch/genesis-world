import torch
import math

n_envs = 10
env_spacing = (3.0, 3.0)
grid_size = int(math.ceil(math.sqrt(n_envs)))

# Replicating Genesis grid layout
origins = []
for i in range(n_envs):
    row = i // grid_size
    col = i % grid_size
    
    # Genesis standard grid offset: does it center or start at 0?
    # Let's print both possibilities to see which matches
    x = col * env_spacing[0]
    y = row * env_spacing[1]
    origins.append((x, y))

print("Genesis Origins if starting at (0,0):", origins)
