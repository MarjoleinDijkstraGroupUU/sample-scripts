import hoomd
import gsd.hoomd
import numpy as np
from itertools import product
from pathlib import Path


def create_system(m: int):
    n_particles = 4 * m**3

    # Particles' positioning
    spacing = 1.5
    # Because this is a 2D system, we use the square root
    K = np.ceil(np.sqrt(n_particles)).astype(np.int64)
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    # Repeat twice for 2D systems
    position = list(product(x, repeat=2))
    position = np.array(position[0:n_particles])
    # The positions are expected to have 3 dimensions, so we need to
    # add a column of zeros for the z-dimension
    position = np.column_stack((position, np.zeros((n_particles,))))
    # "4" is the size of the quaternion for setting the orientation
    orientation = [(1, 0, 0, 0)] * n_particles

    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = n_particles
    snapshot.particles.position = position
    snapshot.particles.orientation = orientation
    # This tells the simulation object that there is only one type of 
    # particles (monodisperse system)
    snapshot.particles.typeid = np.zeros((n_particles,))
    snapshot.particles.types = ["A"]
    # The box can have tilts and different sizes, check the documentation.
    # Here we only use the first two because it is a 2D system
    snapshot.configuration.box = [L, L, 0, 0, 0, 0]

    file_path = Path("lattice.gsd")

    if file_path.exists():
        file_path.unlink()

    with gsd.hoomd.open(name=file_path, mode="xb") as f:
        f.append(snapshot)

    return None


def randomize_system(m: int):
    # Create the initial lattice, it is a square lattice
    create_system(m)

    # Initialize the PRNG and the simulation object
    rng = np.random.default_rng()
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=rng.integers(0, 60_000))
    sim.create_state_from_gsd(filename="lattice.gsd")

    # We define the integrator for hard-spheres
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.2)
    mc.shape["sphere"] = {"diameter": 1.0}
    sim.operations.integrator = mc
    initial_snapshot = sim.state.get_snapshot()

    # Run the system and randomize it
    sim.run(1e4)

    acceptance = mc.translate_moves[0] / sum(mc.translate_moves)
    print(f"Acceptance ratio: {acceptance}")
    # There shouldn't be any rotations
    assert sum(mc.rotate_moves) == 0

    # Save the snapshot of the system
    file_path = Path("random.gsd")

    # Sometimes the GSD writer complains that there is another
    # file with the same name, so we delete it and write the current
    # state again for restarts
    if file_path.exists():
        file_path.unlink()

    hoomd.write.GSD.write(state=sim.state, filename=str(file_path.resolve()), mode="xb")

    return sim


def main():
    # Create and instantiate the system
    m = 4
    sim = randomize_system(m)

    # Compute the volume fraction
    diameter = 1.0
    V_particle = np.pi * diameter**2 / 4.0
    initial_volume_fraction = sim.state.N_particles * V_particle / sim.state.box.volume
    print(f"Initial volume fraction: {initial_volume_fraction}")

    # We are now defining the integrator for hard-spheres
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.2)
    mc.shape["A"] = {"diameter": 1.0}
    # We want to sample the NPT ensemble, so we need to add a box rescaling
    # updater. Here `betaP` is the reduced pressure (check documentation)
    boxmc = hoomd.hpmc.update.BoxMC(trigger=1, betaP=3.0)
    boxl = sim.state.box.Lx
    # These are the parameters that the box updater takes. `standard` means
    # that the volume will get updated as it is, instead of the logarithm
    # of the volume as in conventional implementations of the NPT ensemble
    # (check documentation for all the other available options)
    # `delta` is the maximum volume displacement
    boxmc.volume = {"mode": "standard", "weight": 1.0, "delta": boxl / 4.0}
    sim.operations.integrator = mc
    sim.operations.updaters.append(boxmc)

    # Define the tuners for displacement and box adjustment
    # which would correspond to an NPT ensemble simulation
    # `target` correspond to the target acceptance ratio for each type of move
    eq_time = 750_000
    tune1 = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["d"],
        target=0.4,
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(sim.timestep + eq_time),
            ]
        ),
    )
    sim.operations.tuners.append(tune1)
    tune2 = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
        boxmc=boxmc,
        moves=["volume"],
        target=0.2,
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(sim.timestep + eq_time),
            ]
        ),
    )
    sim.operations.tuners.append(tune2)

    # Run equilibration steps to adjust movements
    sim.run(eq_time)
    # Always remember to remove the tuners so that they are not active
    # during production runs
    sim.operations.tuners.remove(tune1)
    sim.operations.tuners.remove(tune2)

    # Start the production run, collecting information from the system
    logger = hoomd.logging.Logger()
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    sim.operations.computes.append(thermodynamic_properties)
    logger.add(thermodynamic_properties)
    logger.add(mc, quantities=["type_shapes"])

    # We save information every 1000 MC steps
    gsd_writer = hoomd.write.GSD(
        filename="trajectory.gsd",
        trigger=hoomd.trigger.Periodic(1_000),
        mode="xb",
        filter=hoomd.filter.All(),
        log=logger,
    )
    sim.operations.writers.append(gsd_writer)
    sim.run(500_000)
    movement_ratio = mc.translate_moves[0] / sum(mc.translate_moves)
    print(movement_ratio)
    print(boxmc.volume_moves)
    volume_ratio = boxmc.volume_moves[0] / sum(boxmc.volume_moves)
    print(volume_ratio)


if __name__ == "__main__":
    main()
