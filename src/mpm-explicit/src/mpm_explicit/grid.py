from functools import cached_property

import warp as wp


@wp.struct
class Grid:
    """
    A eulerian grid calculated over a set of particles

    Attributes
    """
    min_coord: wp.vec3
    max_coord: wp.vec3
    dimensions: wp.vec3ui
    cell_size: wp.vec3

    masses: wp.array3d[float]
    velocities: wp.array3d[wp.vec3]
    new_velocities: wp.array3d[wp.vec3]
    forces: wp.array3d[wp.vec3]


    @cached_property
    def flat_dimensions(self) -> int:
        return int(self.dimensions[0]) * int(self.dimensions[1]) * int(self.dimensions[2])

    def init(self, min_coord: wp.vec3, max_coord: wp.vec3, dimensions: wp.vec3ui):
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.dimensions = dimensions
        self.cell_size = wp.vec3(
            (max_coord[0] - min_coord[0]) / float(dimensions[0]),
            (max_coord[1] - min_coord[1]) / float(dimensions[1]),
            (max_coord[2] - min_coord[2]) / float(dimensions[2]),
        )

        self.masses = wp.zeros(shape=self.dimensions, dtype=float, device="cuda")
        self.velocities = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")
        self.new_velocities = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")
        self.forces = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")

    def center(self) -> wp.vec3:
        return (self.min_coord + self.max_coord) * 0.5

    def half_dimensions(self) -> wp.vec3:
        return wp.vec3(
            float(self.dimensions[0]) * 0.5,
            float(self.dimensions[1]) * 0.5,
            float(self.dimensions[2]) * 0.5,
        )

    def clear(self):
        self.masses.zero_()
        self.velocities.zero_()
        self.new_velocities.zero_()
        self.forces.zero_()


@wp.func
def grid_index_from_flat(grid: Grid, idx: wp.int32) -> tuple[wp.int32, wp.int32, wp.int32]:
    dim_y = int(grid.dimensions[1])
    dim_z = int(grid.dimensions[2])
    i = idx // (dim_y * dim_z)
    rem = idx % (dim_y * dim_z)
    j = rem // dim_z
    k = rem % dim_z
    return (i, j, k)

@wp.func
def grid_index_to_flat(grid: Grid, i: wp.int32, j: wp.int32, k: wp.int32) -> wp.int32:
    return (i * int(grid.dimensions[1]) + j) * int(grid.dimensions[2]) + k


@wp.func
def grid_index_to_coord(grid: Grid, i: wp.int32, j: wp.int32, k: wp.int32) -> wp.vec3:
    return wp.vec3(
        grid.min_coord[0] + float(i) * grid.cell_size[0],
        grid.min_coord[1] + float(j) * grid.cell_size[1],
        grid.min_coord[2] + float(k) * grid.cell_size[2],
    )
