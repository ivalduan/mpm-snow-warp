from datetime import datetime
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import warp as wp

from mpm_explicit.constants import RENDER_DENSITY_CONTRAST
from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles

RECORDINGS_DIR = Path("recordings")


def init(grid: Grid, obstacles: list[wp.Mesh]):
    rr.init("mpm")
    rr.spawn(connect=False)

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rr.set_sinks(
        rr.GrpcSink(),
        rr.FileSink(RECORDINGS_DIR / f"run_{timestamp}.rrd"),
    )

    min_pos = [float(grid.min_coord[0]), float(grid.min_coord[1]), float(grid.min_coord[2])]
    max_pos = [float(grid.max_coord[0]), float(grid.max_coord[1]), float(grid.max_coord[2])]

    center = [
        0.5 * (min_pos[0] + max_pos[0]),
        0.5 * (min_pos[1] + max_pos[1]),
        0.5 * (min_pos[2] + max_pos[2]),
    ]

    half_sizes = [
        0.5 * (max_pos[0] - min_pos[0]),
        0.5 * (max_pos[1] - min_pos[1]),
        0.5 * (max_pos[2] - min_pos[2]),
    ]

    rr.log(
        "mpm/static/box",
        rr.Boxes3D(
            half_sizes=half_sizes,
            centers=center,
            colors=[128, 128, 128]
        ),
        static=True
    )

    for i, mesh in enumerate(obstacles):
        rr.log(
            f"mpm/static/obstacle_{i}",
            rr.Mesh3D(
                vertex_positions=mesh.points.numpy(),
                triangle_indices=mesh.indices.numpy().reshape(-1, 3),
                vertex_colors=[128, 128, 128, 128]
            ),
            static=True
        )


def density_colors(particles: Particles, contrast: float = RENDER_DENSITY_CONTRAST) -> np.ndarray:
    """Grayscale shading from each particle's current density relative to its
    own reference (construction-time) density, matching the 2D reference's
    render code: particles near their reference density come out white/bright,
    while locally less dense (fluffed-up/fractured) snow shades darker.
    """
    densities = particles.densities.numpy()
    reference_densities = particles.masses.numpy() / particles.volumes.numpy()

    shade = densities / reference_densities * contrast + (1.0 - contrast)
    shade = np.clip(shade, 0.0, 1.0)

    return np.repeat((shade * 255.0).astype(np.uint8)[:, None], 3, axis=1)


def render(t: float, positions: np.ndarray, colors: np.ndarray):
    rr.set_time("step", timestamp=t)
    rr.log("mpm/particles", rr.Points3D(positions=positions, colors=colors))
