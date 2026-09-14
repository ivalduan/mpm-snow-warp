from functools import cached_property

import warp as wp

from mpm_explicit.constants import (
    CFL,
    COULOMB_FRICTION,
    EPSILON,
    EPSILON_SQ,
    GRAVITY,
    MAX_COLLISION_DIST,
    MAX_TIMESTEP,
    MIN_TIMESTEP,
    PICFLIP_ALPHA,
)
from mpm_explicit.grid import (
    Grid,
    grid_index_from_flat,
    grid_index_to_coord,
    grid_index_to_flat,
)
from mpm_explicit.particles import Particles
from mpm_explicit.utils import (
    bspline_dw,
    bspline_w,
    extract_rotation,
    safe_svd3,
)


@wp.struct
class Weights:
    base: wp.array[wp.vec3i]
    wx: wp.array[wp.vec4]
    wy: wp.array[wp.vec4]
    wz: wp.array[wp.vec4]
    dwx: wp.array[wp.vec4]
    dwy: wp.array[wp.vec4]
    dwz: wp.array[wp.vec4]

    def init(self, n: int):
        self.base = wp.empty(n, dtype=wp.vec3i, device="cuda")
        self.wx = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.wy = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.wz = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.dwx = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.dwy = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.dwz = wp.empty(n, dtype=wp.vec4, device="cuda")


class Solver:
    grid: Grid
    particles: Particles
    obstacles: list[wp.Mesh]

    # CFL-style constants driving the adaptive timestep (constant for the
    # solver's lifetime, so they're plain Python floats baked into the
    # captured graph rather than device arrays).
    cfl: float
    max_timestep: float

    t: float = 0

    _weights: Weights

    # Adaptive timestep, recomputed every step from the particles' current
    # max speed. Lives in a device array (rather than being a plain Python
    # float) so it can be updated in place between/within replays of the
    # captured CUDA graph.
    _dt: wp.array[float]
    _max_speed_sq: wp.array[float]
    _cell_size_min: float

    # These need to be signed due to wp.utils.array_scan limitations
    _active_count: wp.array[int]
    _active_flags: wp.array[int]
    _active_offsets: wp.array[int]
    _active_nodes: wp.array[wp.vec3i]

    _first: wp.array[int]
    _graph: wp.Graph

    def __init__(
        self,
        grid: Grid,
        particles: Particles,
        obstacles: list[wp.Mesh],
        cfl: float = CFL,
        max_timestep: float = MAX_TIMESTEP,
        min_timestep: float = MIN_TIMESTEP,
    ):
        self.grid = grid
        self.particles = particles
        self.obstacles = obstacles
        self.cfl = cfl
        self.max_timestep = max_timestep
        self.min_timestep = min_timestep
        self._cell_size_min = min(
            float(grid.cell_size[0]), float(grid.cell_size[1]), float(grid.cell_size[2])
        )

        self._weights = Weights()
        self._weights.init(len(particles))
        self._first = wp.zeros(1, dtype=int)
        self._first.fill_(1)

        self._dt = wp.zeros(shape=1, dtype=float, device="cuda")
        self._dt.fill_(self.max_timestep)
        self._max_speed_sq = wp.zeros(shape=1, dtype=float, device="cuda")

        flat_size = self.grid.flat_dimensions
        self._active_flags = wp.zeros(shape=flat_size, dtype=wp.int32, device="cuda")
        self._active_offsets = wp.zeros(shape=flat_size, dtype=wp.int32, device="cuda")
        self._active_nodes = wp.zeros(shape=flat_size, dtype=wp.vec3i, device="cuda")
        self._active_count = wp.zeros(shape=1, dtype=wp.int32, device="cuda")

        self._graph = self._capture_graph()

    @cached_property
    def obstacle_ids(self) -> wp.array[wp.uint64]:
        return wp.array([obs.id for obs in self.obstacles], dtype=wp.uint64)

    def _capture_graph(self):
        with wp.ScopedCapture(
            device="cuda", capture_mode=wp.CaptureMode.RELAXED
        ) as first_capture:
            # `particles.densities` has already been resampled from the grid
            # earlier this same step (see below), so this just has to turn
            # that first-frame density into a reference volume, once.
            wp.launch(
                kernel=k_set_initial_volumes,
                dim=len(self.particles),
                inputs=[self.particles],
            )

            self._first.fill_(0)

        with wp.ScopedCapture(
            device="cuda", capture_mode=wp.CaptureMode.RELAXED
        ) as capture:
            # adaptive timestep, from the particles' speed at the end of the
            # previous step (or their initial speed, on the very first step)
            self._max_speed_sq.zero_()
            wp.launch(
                kernel=k_compute_max_speed_sq,
                dim=len(self.particles),
                inputs=[self.particles, self._max_speed_sq],
            )
            wp.launch(
                kernel=k_compute_adaptive_dt,
                dim=1,
                inputs=[
                    self._max_speed_sq,
                    self._cell_size_min,
                    self.cfl,
                    self.max_timestep,
                    self.min_timestep,
                    self._dt,
                ],
            )

            # p2g
            wp.launch(
                kernel=k_compute_weights,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights],
            )

            wp.launch(
                kernel=k_p2g,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights],
            )

            # resample each particle's local density from the grid every
            # step (used to seed the reference volume on the first frame,
            # and to drive density-based rendering afterwards)
            wp.launch(
                kernel=k_calculate_density,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, self.particles],
            )

            wp.launch(
                kernel=k_normalize_grid,
                dim=self.grid.dimensions,
                inputs=[self.grid, self._active_flags],
            )

            wp.utils.array_scan(
                self._active_flags, self._active_offsets, inclusive=False
            )

            wp.launch(
                kernel=k_collect_active_nodes,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self._active_flags,
                    self._active_offsets,
                    self._active_nodes,
                    self._active_count,
                ],
            )

            wp.capture_if(self._first, first_capture.graph)

            wp.launch(
                kernel=k_calculate_forces,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, self.particles],
            )

            wp.launch(
                kernel=k_update_grid,
                dim=self.grid.flat_dimensions,
                inputs=[self.grid, self._active_nodes, self._active_count, self._dt],
            )

            wp.launch(
                kernel=k_calculate_grid_collisions,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self._active_nodes,
                    self._active_count,
                    self.obstacle_ids,
                    self._dt,
                ],
            )

            wp.launch(
                kernel=k_update_deformations,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights, self._dt],
            )

            # g2p
            wp.launch(
                kernel=k_g2p,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights],
            )

            # update particles
            wp.launch(
                kernel=k_calculate_particle_collisions,
                dim=len(self.particles),
                inputs=[self.particles, self.obstacle_ids, self._dt],
            )

            wp.launch(
                kernel=k_advect_particle_positions,
                dim=len(self.particles),
                inputs=[self.particles, self._dt],
            )

            # clear
            self.grid.clear()

        return capture.graph

    def update(self):
        wp.capture_launch(self._graph)
        # the timestep is recomputed on-device at the start of the graph, so
        # pull the value that was actually used to advance the host-side clock
        self.t += float(self._dt.numpy()[0])


@wp.kernel
def k_compute_weights(particles: Particles, grid: Grid, weights: Weights):
    p = wp.tid()

    position = particles.positions[p]

    grid_pos = position - grid.min_coord
    i = wp.int32(wp.floor(grid_pos[0] / grid.cell_size[0])) - 1
    j = wp.int32(wp.floor(grid_pos[1] / grid.cell_size[1])) - 1
    k = wp.int32(wp.floor(grid_pos[2] / grid.cell_size[2])) - 1
    weights.base[p] = wp.vec3i(i, j, k)

    wx = wp.vec4(0.0)
    wy = wp.vec4(0.0)
    wz = wp.vec4(0.0)
    dwx = wp.vec4(0.0)
    dwy = wp.vec4(0.0)
    dwz = wp.vec4(0.0)

    for d in range(4):
        nodex = grid.min_coord[0] + float(i + d) * grid.cell_size[0]
        nodey = grid.min_coord[1] + float(j + d) * grid.cell_size[1]
        nodez = grid.min_coord[2] + float(k + d) * grid.cell_size[2]
        x = (position[0] - nodex) / grid.cell_size[0]
        y = (position[1] - nodey) / grid.cell_size[1]
        z = (position[2] - nodez) / grid.cell_size[2]
        wx[d] = bspline_w(x)
        wy[d] = bspline_w(y)
        wz[d] = bspline_w(z)
        dwx[d] = bspline_dw(x) / grid.cell_size[0]
        dwy[d] = bspline_dw(y) / grid.cell_size[1]
        dwz[d] = bspline_dw(z) / grid.cell_size[2]

    weights.wx[p] = wx
    weights.wy[p] = wy
    weights.wz[p] = wz
    weights.dwx[p] = dwx
    weights.dwy[p] = dwy
    weights.dwz[p] = dwz


@wp.kernel
def k_p2g(particles: Particles, grid: Grid, weights: Weights):
    p = wp.tid()

    mass = particles.masses[p]
    velocity = particles.velocities[p]

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                    i < 0
                    or i >= grid.dimensions[0]
                    or j < 0
                    or j >= grid.dimensions[1]
                    or k < 0
                    or k >= grid.dimensions[2]
                ):
                    continue

                weight = wx[di] * wy[dj] * wz[dk]

                wp.atomic_add(grid.masses, i, j, k, weight * mass)
                wp.atomic_add(grid.velocities, i, j, k, weight * mass * velocity)


@wp.kernel
def k_normalize_grid(grid: Grid, active_flags: wp.array[int]):
    i, j, k = wp.tid()

    mass = grid.masses[i, j, k]
    idx = grid_index_to_flat(grid, i, j, k)

    if mass > EPSILON:
        grid.velocities[i, j, k] = grid.velocities[i, j, k] / mass
        active_flags[idx] = 1
    else:
        grid.velocities[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        active_flags[idx] = 0


@wp.kernel
def k_collect_active_nodes(
    grid: Grid,
    active_flags: wp.array[int],
    active_offsets: wp.array[int],
    active_nodes: wp.array[wp.vec3i],
    active_count: wp.array[int],
):
    flat_idx = wp.tid()
    dim_x = int(grid.dimensions[0])
    dim_y = int(grid.dimensions[1])
    dim_z = int(grid.dimensions[2])
    total_elements = dim_x * dim_y * dim_z

    if active_flags[flat_idx] == 1:
        offset = active_offsets[flat_idx]

        (i, j, k) = grid_index_from_flat(grid, flat_idx)

        active_nodes[offset] = wp.vec3i(i, j, k)

    if flat_idx == total_elements - 1:
        active_count[0] = active_offsets[flat_idx] + active_flags[flat_idx]


@wp.kernel
def k_calculate_density(grid: Grid, weights: Weights, particles: Particles):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    density = float(0.0)
    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
                    or k < 0
                    or k >= grid.masses.shape[2]
                ):
                    continue

                weight = wx[di] * wy[dj] * wz[dk]
                density += (
                    weight
                    * grid.masses[i, j, k]
                    / (grid.cell_size[0] * grid.cell_size[1] * grid.cell_size[2])
                )

    particles.densities[p] = density


@wp.kernel
def k_set_initial_volumes(particles: Particles):
    p = wp.tid()

    particles.volumes[p] = particles.masses[p] / particles.densities[p]


@wp.kernel
def k_compute_max_speed_sq(particles: Particles, max_speed_sq: wp.array(dtype=float)):
    p = wp.tid()
    v = particles.velocities[p]
    wp.atomic_max(max_speed_sq, 0, wp.dot(v, v))


@wp.kernel
def k_compute_adaptive_dt(
    max_speed_sq: wp.array(dtype=float),
    cell_size_min: float,
    cfl: float,
    max_timestep: float,
    min_timestep: float,
    dt: wp.array(dtype=float),
):
    # dt = min(CFL * min_cell_size / max_speed, MAX_TIMESTEP); when the
    # particles are (almost) at rest this saturates at max_timestep.
    speed = wp.sqrt(wp.max(max_speed_sq[0], EPSILON_SQ))
    dt[0] = wp.max(min_timestep, wp.min(cfl * cell_size_min / speed, max_timestep))


@wp.func
def mu_(F_P: wp.mat33, xi: float, mu_0: float) -> float:
    J_P = wp.determinant(F_P)
    return mu_0 * wp.exp(xi * (1.0 - J_P))


@wp.func
def lambda_(F_P: wp.mat33, xi: float, lambda_0: float) -> float:
    J_P = wp.determinant(F_P)
    return lambda_0 * wp.exp(xi * (1.0 - J_P))


@wp.kernel
def k_calculate_forces(grid: Grid, weights: Weights, particles: Particles):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    dwx = weights.dwx[p]
    dwy = weights.dwy[p]
    dwz = weights.dwz[p]

    F_Ep = particles.F_Es[p]
    F_Pp = particles.F_Ps[p]
    xi = particles.xis[p]
    mu_0 = particles.mus[p]
    mu = mu_(F_Pp, xi, mu_0)
    lambda_0 = particles.lambdas[p]
    lmbd = lambda_(F_Pp, xi, lambda_0)

    R_Ep = extract_rotation(F_Ep)
    J_Ep = wp.determinant(F_Ep)

    V_0 = particles.volumes[p]

    I = wp.identity(3, dtype=wp.float32)
    pk_stress = (2.0 * mu) * (F_Ep - R_Ep) * wp.transpose(F_Ep) + (
        lmbd * J_Ep * (J_Ep - 1.0)
    ) * I
    force = (-1.0 * V_0) * pk_stress

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
                    or k < 0
                    or k >= grid.masses.shape[2]
                ):
                    continue

                weight_grad = wp.vec3(
                    dwx[di] * wy[dj] * wz[dk],
                    wx[di] * dwy[dj] * wz[dk],
                    wx[di] * wy[dj] * dwz[dk],
                )

                wp.atomic_add(grid.forces, i, j, k, force @ weight_grad)


@wp.kernel
def k_update_grid(
    grid: Grid,
    active_nodes: wp.array[wp.vec3i],
    active_count: wp.array[int],
    dt: wp.array(dtype=float),
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j, k = coord[0], coord[1], coord[2]

    m = grid.masses[i, j, k]
    v = grid.velocities[i, j, k]

    v += dt[0] * (GRAVITY + grid.forces[i, j, k] / m)

    grid.new_velocities[i, j, k] = v


@wp.kernel
def k_calculate_grid_collisions(
    grid: Grid,
    active_nodes: wp.array[wp.vec3i],
    active_count: wp.array[int],
    obstacles: wp.array[wp.uint64],
    dt: wp.array(dtype=float),
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j, k = coord[0], coord[1], coord[2]

    position = grid_index_to_coord(grid, i, j, k)
    velocity = grid.new_velocities[i, j, k]
    dt_val = dt[0]

    for b in range(obstacles.shape[0]):
        test_position = position + dt_val * velocity
        obs_id = obstacles[b]

        query = wp.mesh_query_point_sign_normal(
            obs_id, test_position, MAX_COLLISION_DIST, EPSILON
        )

        if query.result:
            p = wp.mesh_eval_position(obs_id, query.face, query.u, query.v)
            delta = test_position - p
            dist = wp.length(delta) * query.sign

            if dist <= 0.0:
                delta_len = wp.length(delta)

                if delta_len > 1e-6:
                    normal = (delta / delta_len) * query.sign
                else:
                    # Fallback if exactly on the face boundary: calculate normal from vertices
                    v0 = wp.mesh_eval_position(obs_id, query.face, 0.0, 0.0)
                    v1 = wp.mesh_eval_position(obs_id, query.face, 1.0, 0.0)
                    v2 = wp.mesh_eval_position(obs_id, query.face, 0.0, 1.0)
                    normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))

                velocity_normal = wp.dot(velocity, normal)
                velocity_tangent = velocity - velocity_normal * normal

                if velocity_normal > 0.0:
                    continue

                if wp.length(velocity_tangent) <= -COULOMB_FRICTION * velocity_normal:
                    velocity = wp.vec3(0.0, 0.0, 0.0)
                else:
                    velocity = velocity_tangent + COULOMB_FRICTION * velocity_normal * (
                        velocity_tangent / wp.length(velocity_tangent)
                    )

    grid.new_velocities[i, j, k] = velocity


@wp.kernel
def k_update_deformations(
    particles: Particles, grid: Grid, weights: Weights, dt: wp.array(dtype=float)
):
    p = wp.tid()

    F_E = particles.F_Es[p]
    F_P = particles.F_Ps[p]

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]
    dwx = weights.dwx[p]
    dwy = weights.dwy[p]
    dwz = weights.dwz[p]

    vel_grad = wp.mat33(0.0)

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                    i < 0
                    or i >= grid.dimensions[0]
                    or j < 0
                    or j >= grid.dimensions[1]
                    or k < 0
                    or k >= grid.dimensions[2]
                ):
                    continue

                weight_grad = wp.vec3(
                    dwx[di] * wy[dj] * wz[dk],
                    wx[di] * dwy[dj] * wz[dk],
                    wx[di] * wy[dj] * dwz[dk],
                )

                new_velocity = grid.new_velocities[i, j, k]

                vel_grad += wp.outer(new_velocity, weight_grad)

    I = wp.identity(3, dtype=wp.float32)
    F_E = (I + dt[0] * vel_grad) * F_E
    F = F_E * F_P

    U, s, V = safe_svd3(F_E)

    lo = 1.0 - particles.thetas_c[p]
    hi = 1.0 + particles.thetas_s[p]
    for i in range(3):
        s[i] = wp.clamp(s[i], lo, hi)

    s_inv = wp.vec3(1.0 / s[0], 1.0 / s[1], 1.0 / s[2])

    particles.F_Es[p] = U * wp.diag(s) * wp.transpose(V)
    particles.F_Ps[p] = V * wp.diag(s_inv) * wp.transpose(U) * F


@wp.kernel
def k_g2p(particles: Particles, grid: Grid, weights: Weights):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    v_pic = wp.vec3(0.0)
    v_flip = particles.velocities[p]

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
                    or k < 0
                    or k >= grid.masses.shape[2]
                ):
                    continue

                weight = wx[di] * wy[dj] * wz[dk]
                node_velocity = grid.velocities[i, j, k]
                node_new_velocity = grid.new_velocities[i, j, k]

                v_pic += node_new_velocity * weight
                v_flip += (node_new_velocity - node_velocity) * weight

    particles.velocities[p] = (1.0 - PICFLIP_ALPHA) * v_pic + PICFLIP_ALPHA * v_flip


@wp.kernel
def k_calculate_particle_collisions(
    particles: Particles, boundaries: wp.array[wp.uint64], dt: wp.array(dtype=float)
):
    p = wp.tid()
    dt_val = dt[0]

    position = particles.positions[p]
    velocity = particles.velocities[p]

    for b in range(boundaries.shape[0]):
        test_position = position + dt_val * velocity
        boundary_id = boundaries[b]

        query = wp.mesh_query_point_sign_normal(
            boundary_id, test_position, MAX_COLLISION_DIST, EPSILON
        )

        if query.result:
            closest_p = wp.mesh_eval_position(boundary_id, query.face, query.u, query.v)
            delta = test_position - closest_p
            dist = wp.length(delta) * query.sign

            if dist <= 0.0:
                delta_len = wp.length(delta)
                normal = wp.vec3(0.0, 0.0, 0.0)

                if delta_len > 1e-6:
                    normal = (delta / delta_len) * query.sign
                else:
                    # Fallback for boundary touch
                    v0 = wp.mesh_eval_position(boundary_id, query.face, 0.0, 0.0)
                    v1 = wp.mesh_eval_position(boundary_id, query.face, 1.0, 0.0)
                    v2 = wp.mesh_eval_position(boundary_id, query.face, 0.0, 1.0)
                    normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))

                velocity_normal = wp.dot(velocity, normal)
                velocity_tangent = velocity - velocity_normal * normal

                if velocity_normal > 0.0:
                    continue

                if wp.length(velocity_tangent) <= -COULOMB_FRICTION * velocity_normal:
                    velocity = wp.vec3(0.0, 0.0, 0.0)
                else:
                    velocity = velocity_tangent + COULOMB_FRICTION * velocity_normal * (
                        velocity_tangent / wp.length(velocity_tangent)
                    )

    particles.velocities[p] = velocity


@wp.kernel
def k_advect_particle_positions(particles: Particles, dt: wp.array(dtype=float)):
    p = wp.tid()
    particles.positions[p] += dt[0] * particles.velocities[p]
