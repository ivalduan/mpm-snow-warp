import numpy as np
import trimesh
import warp as wp
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

import mpm_explicit.renderer as rd
from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles, sphere_particle_count
from mpm_explicit.solver import Solver

DURATION = 4.0
FPS = 60


def setup0(grid, particles, obstacles):
    max_coord = (1.0, 1.0, 3.0)
    min_coord = (-1.0, -1.0, 1.0)
    dimensions = (128, 128, 128)
    cell_size = tuple(
        (hi - lo) / n
        for hi, lo, n in zip(max_coord, min_coord, dimensions, strict=True)
    )

    grid.init(
        min_coord=wp.vec3(*min_coord),
        max_coord=wp.vec3(*max_coord),
        dimensions=wp.vec3ui(
            wp.uint32(dimensions[0]),
            wp.uint32(dimensions[1]),
            wp.uint32(dimensions[2]),
        ),
    )

    particles.sample_sphere(
        center=wp.vec3(0.0, 0.0, 2.7),
        radius=0.2,
        particle_diam=cell_size[0] * 0.5,
    )
    particles.velocities.fill_(wp.vec3(0.0, 0.0, -3.0))

    mesh_files = [
        "assets/models/floor.obj",
        "assets/models/diamond.obj",
    ]

    for f in mesh_files:
        tm = trimesh.load_mesh(f)
        rotation = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        tm.apply_transform(rotation)
        points = np.asarray(tm.vertices, dtype=np.float32)
        indices = np.asarray(tm.faces, dtype=np.int32)
        mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device="cuda"),
            indices=wp.array(indices.reshape(-1), dtype=wp.int32, device="cuda"),
        )
        obstacles.append(mesh)


def setup1(grid, particles, _obstacles):
    max_coord = (1.0, 2.0, 2.0)
    min_coord = (-1.0, 0.0, 0.0)
    dimensions = (128, 128, 128)

    grid.init(
        min_coord=wp.vec3(*min_coord),
        max_coord=wp.vec3(*max_coord),
        dimensions=wp.vec3ui(
            wp.uint32(dimensions[0]),
            wp.uint32(dimensions[1]),
            wp.uint32(dimensions[2]),
        ),
    )

    cell_size = tuple(
        (hi - lo) / n
        for hi, lo, n in zip(max_coord, min_coord, dimensions, strict=True)
    )

    particle_diam = cell_size[0] * 0.5

    # Two spheres on either side of the domain's x-center, launched toward
    # each other so they collide mid-flight.
    emitters = [
        {
            "center": wp.vec3(0.0, 1.7, 1.7),
            "radius": 0.15,
            "velocity": wp.vec3(0.0, -2.0, -2.0),
        },
        {
            "center": wp.vec3(0.0, 1.7, 0.3),
            "radius": 0.2,
            "velocity": wp.vec3(0.0, -3.0, 3.0),
        },
        {
            "center": wp.vec3(0.0, 0.37, 1.75),
            "radius": 0.2,
            "velocity": wp.vec3(0.0, 3.0, -2.0),
        },
    ]

    total_particles = sum(
        sphere_particle_count(e["radius"], particle_diam) for e in emitters
    )
    particles.init(total_particles)

    offset = 0
    for emitter in emitters:
        offset += particles.fill_sphere(
            offset=offset,
            center=emitter["center"],
            radius=emitter["radius"],
            particle_diam=particle_diam,
            velocity=emitter["velocity"],
        )

    particles.fill_deformations()


def main():
    print("Initializing warp and compiling kernels")
    wp.init()

    grid = Grid()
    particles = Particles()
    obstacles = []

    # setup0(grid=grid, particles=particles, obstacles=obstacles);
    setup1(grid=grid, particles=particles, _obstacles=obstacles);

    # dt is adaptive (CFL-based, recomputed every step from the particles'
    # current speed) rather than fixed, so the number of steps needed to
    # reach DURATION isn't known ahead of time - only the frame count is.
    solver = Solver(grid, particles, obstacles)

    rd.init(grid, obstacles)
    rd.render(solver.t, solver.particles.positions.numpy(), rd.density_colors(solver.particles))

    frame_duration = 1.0 / FPS
    next_frame_time = 0.0
    total_frames = round(DURATION * FPS)

    class TimePerStepColumn(ProgressColumn):
        """Calculates and displays the average time taken per step."""

        def __init__(self, moving_average=True):
            super().__init__()
            # If True, calculates the recent average using 1 / speed.
            # If False, calculates the cumulative average since the start.
            self.moving_average = moving_average

        def render(self, task):
            if self.moving_average:
                speed = task.speed
                if not speed or speed <= 0:
                    return Text("? s/step", style="dim")
                time_per_step = 1.0 / speed
            else:
                if task.completed == 0 or task.elapsed is None:
                    return Text("? s/step", style="dim")
                time_per_step = task.elapsed / task.completed

            if time_per_step < 1.0:
                return Text(f"{time_per_step * 1000:.1f} masses/step", style="cyan")
            else:
                return Text(f"{time_per_step:.2f} s/step", style="cyan")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        MofNCompleteColumn(),
        "•",
        TimePerStepColumn(moving_average=False),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        # Step count isn't known in advance with an adaptive dt, so "Steps"
        # is tracked as an indeterminate counter; "Frames" still has a fixed
        # total since it's driven by simulated time, not step count.
        task_steps = progress.add_task(description="Steps", total=None)
        task_frames = progress.add_task(description="Frames", total=total_frames)
        while solver.t < DURATION:
            solver.update()
            if solver.t >= next_frame_time:
                rd.render(solver.t, solver.particles.positions.numpy(), rd.density_colors(solver.particles))
                next_frame_time += frame_duration
                progress.update(task_frames, advance=1)

            progress.update(task_steps, advance=1)
