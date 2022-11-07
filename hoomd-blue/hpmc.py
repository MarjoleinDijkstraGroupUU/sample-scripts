import hoomd
import gsd.hoomd
import numpy as np
import itertools
from pathlib import Path


def create_system(m: int):
    n_particles = 4 * m**3
    print(f"Number of particles: {n_particles}")

    # Particles' positioning
    spacing = 2.0
    # Use the cubic root because this is a 3D system
    K = np.ceil(np.cbrt(n_particles)).astype(np.int64)
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    # Repeat three times once again for a 3D system
    position = list(itertools.product(x, repeat=3))
    position = position[0:n_particles]
    # "4" is the size of the quaternion for setting the orientation
    orientation = [(1, 0, 0, 0)] * n_particles

    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = n_particles
    snapshot.particles.position = position
    snapshot.particles.orientation = orientation
    # This tells the system there is a monodisperse simulation
    snapshot.particles.typeid = np.zeros((n_particles,))
    snapshot.particles.types = ["A"]
    snapshot.configuration.box = [L, L, L, 0, 0, 0]

    file_path = Path("lattice.gsd")

    if file_path.exists():
        file_path.unlink()

    with gsd.hoomd.open(name=file_path, mode="xb") as f:
        f.append(snapshot)

    return None


def randomize_system(m):
    # Initialize the system in a square lattice
    create_system(m)

    # Create the simulation object and set a random number
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=35)
    sim.create_state_from_gsd(filename="lattice.gsd")

    # Define the hard-sphere integrator
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape["A"] = {"diameter": 1.0}
    sim.operations.integrator = mc

    # Run the system to randomize it
    sim.run(10e4)

    acceptance = mc.translate_moves[0] / sum(mc.translate_moves)
    print(f"Acceptance ratio: {acceptance}")
    # There shouldn't be any rotations in this very simple system
    assert sum(mc.rotate_moves) == 0

    file_path = Path("random.gsd")
    if file_path.exists():
        file_path.unlink()

    hoomd.write.GSD.write(state=sim.state, filename=str(file_path.resolve()), mode="xb")


def main():
    # Create and instantiate the system
    m = 4
    randomize_system(m)

    # Load the simulated system
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=35)
    sim.create_state_from_gsd(filename="random.gsd")

    # Compute the volume fraction
    diameter = 1.0
    V_particle = np.pi * diameter**3 / 6.0
    initial_volume_fraction = sim.state.N_particles * V_particle / sim.state.box.volume
    print(f"Initial packing fraction: {initial_volume_fraction}")

    # Create two boxes, the original one and a new one with a target
    # packing fraction
    initial_box = sim.state.box
    final_box = hoomd.Box.from_box(initial_box)
    final_volume_fraction = 0.58
    final_box.volume = sim.state.N_particles * V_particle / final_volume_fraction

    # Define the hard-sphere integrator
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape["A"] = {"diameter": diameter}
    sim.operations.integrator = mc
    periodic = hoomd.trigger.Periodic(10)
    # We use a `QuickCompress` updater to compress the box to the target
    # packing fraction
    compress = hoomd.hpmc.update.QuickCompress(trigger=periodic, target_box=final_box)
    sim.operations.updaters.append(compress)
    while not compress.complete and sim.timestep < 1e6:
        sim.run(1000)

    final_packing = sim.state.N_particles * V_particle / sim.state.box.volume
    print(f"Final packing fraction: {final_packing}")
    hoomd.write.GSD.write(state=sim.state, mode="xb", filename="compressed.gsd")

    # Equilibrate the system
    eq_timesteps = 500_000
    # Add a displacement tuner with a target acceptance ratio of 0.4
    tune = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["d"],
        target=0.4,
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(sim.timestep + eq_timesteps),
            ]
        ),
    )
    sim.operations.tuners.append(tune)
    print("Starting equilibration run...")
    sim.run(eq_timesteps)
    # Always remove the tuner
    sim.operations.tuners.remove(tune)
    # Check the acceptance ratio, should be close to the target ratio
    translate_moves = mc.translate_moves
    print(
        f"Traslational acceptance ratio: {mc.translate_moves[0] / sum(translate_moves)}"
    )

    # Finally, do a production run, gathering thermodynamic information
    logger = hoomd.logging.Logger()
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    sim.operations.computes.append(thermodynamic_properties)
    logger.add(thermodynamic_properties)
    gsd_writer = hoomd.write.GSD(
        filename="trajectory.gsd", trigger=hoomd.trigger.Periodic(1000), mode="xb"
    )
    sim.operations.writers.append(gsd_writer)
    print("Starting production run...")
    sim.run(1_000_000)


if __name__ == "__main__":
    main()
