import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np

traj = gsd.hoomd.open("trajectory.gsd", "rb")
timestep = []
density = []
num_particles = 256

for frame in traj:
    timestep.append(frame.configuration.step)
    new_density = (
        num_particles / frame.log["md/compute/ThermodynamicQuantities/volume"][0]
    )
    density.append(new_density)

print(np.mean(np.array(density)))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(timestep, density)
ax.set_xlabel("timestep")
ax.set_ylabel("density")
plt.show()
