import warp as wp
import math

from mpm_explicit.constants import DEFAULT_THETA_C, DEFAULT_THETA_S, \
    DEFAULT_XI, DEFAULT_E, DEFAULT_NU, DEFAULT_DENSITY


def sphere_particle_count(radius: float, particle_diam: float) -> int:
    """Number of particles `sample_sphere`/`fill_sphere` will generate for a
    sphere of this size, useful to size a shared buffer for multiple
    emitters ahead of time."""
    volume = (4.0 / 3.0) * math.pi * (radius ** 3)
    particle_volume = particle_diam ** 3
    return int(volume / particle_volume)


@wp.struct
class Particle:
    mass: float
    position: wp.vec3
    velocity: wp.vec3
    volume: float
    F_E: wp.mat33
    F_P: wp.mat33

    E: float = DEFAULT_E
    nu: float = DEFAULT_NU
    theta_c: float = DEFAULT_THETA_C
    theta_s: float = DEFAULT_THETA_S
    xi: float = DEFAULT_XI


@wp.struct
class Particles:
    """
    Structure containing the data for all particles

    Attributes:
        volumes (float, constant): The initial volume of the particles, assigned on the first iteration
        masses (float, constant): The mass of the particles
        thetas_c (float, constant): The compression threshold
        thetas_s (float, constant): The stretch threshold
        xis (float, constant): A coefficient that defines how fast the material breaks once yielding
        initial_young_moduli (float, constant): The overall stiffness of the material
        poisson_ratios (float, constant): The Poisson's ratio used to construct the Lamé parameters
        positions (wp.vec3): The current position of the particle
        velocities (wp.vec3): The current velocity of the particle
        F_E (wp.mat33): The elastic part of the particle's current deformation gradient
        F_P (wp.mat33): The plastic part of the particle's current deformation gradient
        densities (float): The particle's current local density, resampled from the grid every
            step (used to set the initial volume on the first frame, and for density-based rendering)
    """
    masses: wp.array[float]
    volumes: wp.array[float]
    densities: wp.array[float]
    positions: wp.array[wp.vec3]
    velocities: wp.array[wp.vec3]
    F_Es: wp.array[wp.mat33]
    F_Ps: wp.array[wp.mat33]

    # This could all be per-material constants
    lambdas: wp.array[float]
    mus: wp.array[float]
    thetas_c: wp.array[float]
    thetas_s: wp.array[float]
    xis: wp.array[float]

    def init(self, n: int):
        self.masses = wp.empty(shape=n, dtype=wp.float32, device="cuda")
        self.volumes = wp.empty(shape=n, dtype=wp.float32, device="cuda")
        self.densities = wp.zeros(shape=n, dtype=wp.float32, device="cuda")
        self.positions = wp.empty(shape=n, dtype=wp.vec3, device="cuda")
        self.velocities = wp.empty(shape=n, dtype=wp.vec3, device="cuda")
        self.F_Es = wp.empty(shape=n, dtype=wp.mat33, device="cuda")
        self.F_Ps = wp.empty(shape=n, dtype=wp.mat33, device="cuda")
        self.lambdas = wp.empty(shape=n, dtype=wp.float32, device="cuda")
        self.mus = wp.empty(shape=n, dtype=wp.float32, device="cuda")
        self.thetas_c = wp.empty(shape=n, dtype=wp.float32, device="cuda")
        self.thetas_s = wp.empty(shape=n, dtype=wp.float32, device="cuda")
        self.xis = wp.empty(shape=n, dtype=wp.float32, device="cuda")

    # def set_particles(self, plist: list[Particle]):
    #     self.init(len(plist))
    #
    #     wp.launch(
    #         kernel=k_particles_fill_from_list,
    #         dim=len(plist),
    #         inputs=[wp.array(plist, dtype=Particle), self],
    #     )
    #
    #     wp.launch(
    #         kernel=k_particles_fill_deformations,
    #         dim=len(self),
    #         inputs=[self]
    #     )

    # def sample_cube(
    #         self,
    #         min_coord: wp.vec3,
    #         cell_size: wp.vec3,
    #         dimensions: wp.vec3,
    #         particles_per_cell: int = 8,
    #         density: float = 400.0,
    #         theta_c: float = DEFAULT_THETA_C,
    #         theta_s: float = DEFAULT_THETA_S,
    #         xi: float = DEFAULT_XI,
    #         E: float = DEFAULT_E,
    #         nu: float = DEFAULT_NU,
    #         seed: int = 0
    # ):
    #     num_particles = int(dimensions[0]) * int(dimensions[1]) * int(dimensions[2]) * particles_per_cell
    #
    #     particle_volume = (cell_size[0] * cell_size[1] * cell_size[2]) / particles_per_cell
    #     particle_mass = density * particle_volume
    #
    #     self.init(num_particles)
    #
    #     self.volumes.fill_(particle_volume)
    #     self.masses.fill_(particle_mass)
    #     self.thetas_c.fill_(theta_c)
    #     self.thetas_s.fill_(theta_s)
    #     self.xis.fill_(xi)
    #     self.mus.fill_(E / (2.0 * (1.0 + nu)))
    #     self.lambdas.fill_(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    #     self.velocities.zero_()
    #
    #     wp.launch(
    #         kernel=k_particles_sample_cube,
    #         dim=(dimensions[0], dimensions[1], dimensions[2], particles_per_cell),
    #         inputs=[min_coord, cell_size, dimensions[1], dimensions[2], particles_per_cell, seed, self.positions],
    #     )
    #
    #     wp.launch(
    #         kernel=k_particles_fill_deformations,
    #         dim=len(self),
    #         inputs=[self]
    #     )

    def sample_sphere(
            self,
            center: wp.vec3,
            radius: float,
            particle_diam: float,
            velocity: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
            material_density: float = DEFAULT_DENSITY,
            theta_c: float = DEFAULT_THETA_C,
            theta_s: float = DEFAULT_THETA_S,
            xi: float = DEFAULT_XI,
            E: float = DEFAULT_E,
            nu: float = DEFAULT_NU,
            seed: int = 0
    ):
        """Allocates a fresh buffer sized for a single sphere and fills it.

        For multiple emitters (e.g. several spheres with different
        velocities sharing one buffer) use `init()` + `fill_sphere()` for
        each emitter instead, so the buffer isn't reallocated/wiped between
        calls.
        """
        num_particles = sphere_particle_count(radius, particle_diam)

        self.init(num_particles)

        self.fill_sphere(
            offset=0,
            center=center,
            radius=radius,
            particle_diam=particle_diam,
            velocity=velocity,
            material_density=material_density,
            theta_c=theta_c,
            theta_s=theta_s,
            xi=xi,
            E=E,
            nu=nu,
            seed=seed,
        )

        self.fill_deformations()

    def fill_sphere(
            self,
            offset: int,
            center: wp.vec3,
            radius: float,
            particle_diam: float,
            velocity: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
            material_density: float = DEFAULT_DENSITY,
            theta_c: float = DEFAULT_THETA_C,
            theta_s: float = DEFAULT_THETA_S,
            xi: float = DEFAULT_XI,
            E: float = DEFAULT_E,
            nu: float = DEFAULT_NU,
            seed: int = 0
    ) -> int:
        num_particles = sphere_particle_count(radius, particle_diam)
        end = offset + num_particles

        particle_volume = particle_diam ** 3
        particle_mass = material_density * particle_volume
        lambda0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu0 = E / (2.0 * (1.0 + nu))

        self.masses[offset:end].fill_(particle_mass)
        self.volumes[offset:end].fill_(particle_volume)
        self.velocities[offset:end].fill_(velocity)

        self.lambdas[offset:end].fill_(lambda0)
        self.mus[offset:end].fill_(mu0)
        self.thetas_c[offset:end].fill_(theta_c)
        self.thetas_s[offset:end].fill_(theta_s)
        self.xis[offset:end].fill_(xi)

        wp.launch(
            kernel=k_particles_sample_sphere,
            dim=num_particles,
            inputs=[center, radius, seed, self.positions[offset:end]],
        )

        return num_particles

    def fill_deformations(self):
        wp.launch(
            kernel=k_particles_fill_deformations,
            dim=len(self),
            inputs=[self]
        )

    # def sample_packed_snowball(
    #         self,
    #         center: wp.vec3,
    #         radius: float,
    #         particle_diam: float,
    #         material_density: float = DEFAULT_DENSITY,
    #         theta_c: float = DEFAULT_THETA_C,
    #         theta_s: float = DEFAULT_THETA_S,
    #         xi: float = DEFAULT_XI,
    #         E: float = DEFAULT_E,
    #         nu: float = DEFAULT_NU,
    #         seed: int = 0,
    #         mass_outer_mult: float = 2.0,
    #         stiffness_outer_mult: float = 5.0,
    #         noise_amplitude: float = 0.4,
    #         noise_frequency: float = 20.0
    # ):
    #     volume = (4.0 / 3.0) * math.pi * (radius ** 3)
    #     particle_volume = particle_diam ** 3
    #     num_particles = int(volume / particle_volume)
    #
    #     self.init(num_particles)
    #
    #     self.volumes.fill_(particle_volume)
    #     self.thetas_c.fill_(theta_c)
    #     self.thetas_s.fill_(theta_s)
    #     self.xis.fill_(xi)
    #     self.velocities.zero_()
    #
    #     wp.launch(
    #         kernel=k_particles_sample_packed_snowball,
    #         dim=num_particles,
    #         inputs=[
    #             center,
    #             radius,
    #             seed,
    #             particle_volume,
    #             material_density,
    #             E,
    #             nu,
    #             mass_outer_mult,
    #             stiffness_outer_mult,
    #             noise_amplitude,
    #             noise_frequency,
    #             self
    #         ],
    #     )
    #
    #     wp.launch(
    #         kernel=k_particles_fill_deformations,
    #         dim=len(self),
    #         inputs=[self]
    #     )

    def __len__(self):
        return len(self.volumes)


# @wp.kernel
# def k_particles_fill_from_list(plist: wp.array[Particle], particles: Particles):
#     i = wp.tid()
#     p = plist[i]
#
#     particles.volumes[i] = p.volume
#     particles.masses[i] = p.mass
#     particles.thetas_c[i] = p.theta_c
#     particles.thetas_s[i] = p.theta_s
#     particles.xis[i] = p.xi
#     particles.mus[i] = p.E / (2.0 * (1.0 + p.nu))
#     particles.lambdas[i] = p.E * p.nu / (
#                 (1.0 + p.nu) * (1.0 - 2.0 * p.nu))
#     particles.positions[i] = p.position
#     particles.velocities[i] = p.velocity
#     particles.F_Es[i] = p.F_e
#     particles.F_Ps[i] = p.F_p


# @wp.kernel
# def k_particles_sample_cube(
#         min_coord: wp.vec3,
#         cell_size: wp.vec3,
#         dim_y: int,
#         dim_z: int,
#         particles_per_cell: int,
#         seed: wp.int32,
#         positions: wp.array[wp.vec3],
# ):
#     i, j, k, p = wp.tid()
#     idx = ((i * dim_y + j) * dim_z + k) * particles_per_cell + p
#
#     state = wp.rand_init(seed, idx)
#     jitter = wp.vec3(wp.randf(state) * cell_size[0], wp.randf(state) * cell_size[1], wp.randf(state) * cell_size[2])
#     cell_origin = min_coord + wp.vec3(float(i) * cell_size[0], float(j) * cell_size[1], float(k) * cell_size[2])
#
#     positions[idx] = cell_origin + jitter


@wp.kernel
def k_particles_sample_sphere(
        center: wp.vec3,
        radius: float,
        seed: wp.int32,
        positions: wp.array[wp.vec3],
):
    p = wp.tid()

    state = wp.rand_init(seed, p)

    u = wp.randf(state)
    v = wp.randf(state)
    w = wp.randf(state)

    phi = 2.0 * 3.1415926535 * u
    cos_theta = 2.0 * v - 1.0
    sin_theta = wp.sqrt(1.0 - cos_theta * cos_theta)
    r = radius * wp.pow(w, 1.0 / 3.0)

    offset = wp.vec3(
        r * sin_theta * wp.cos(phi),
        r * sin_theta * wp.sin(phi),
        r * cos_theta
    )

    positions[p] = center + offset


# @wp.kernel
# def k_particles_sample_packed_snowball(
#         center: wp.vec3,
#         radius: float,
#         seed: wp.int32,
#         particle_volume: float,
#         base_density: float,
#         base_young: float,
#         nu: float,
#         mass_outer_mult: float,
#         stiffness_outer_mult: float,
#         noise_amplitude: float,
#         noise_frequency: float,
#         particles: Particles,
# ):
#     idx = wp.tid()
#
#     state = wp.rand_init(seed, idx)
#
#     u = wp.randf(state)
#     v = wp.randf(state)
#     w = wp.randf(state)
#
#     phi = 2.0 * 3.1415926535 * u
#     cos_theta = 2.0 * v - 1.0
#     sin_theta = wp.sqrt(1.0 - cos_theta * cos_theta)
#
#     r = radius * wp.pow(w, 1.0 / 3.0)
#
#     offset = wp.vec3(
#         r * sin_theta * wp.cos(phi),
#         r * sin_theta * wp.sin(phi),
#         r * cos_theta
#     )
#
#     particles.positions[idx] = center + offset
#
#     normalized_r = r / radius
#     density_mult = wp.lerp(1.0, mass_outer_mult, normalized_r)
#     local_density = base_density * density_mult
#     particles.masses[idx] = local_density * particle_volume
#
#     stiffness_mult = wp.lerp(1.0, stiffness_outer_mult, normalized_r)
#
#     noise_seed = wp.uint32(seed)
#     noise_val = wp.noise(noise_seed, offset * noise_frequency)
#
#     young = base_young * stiffness_mult * (1.0 + noise_amplitude * noise_val)
#     young = wp.max(0.01 * base_young, young)
#
#     mu = young / (2.0 * (1.0 + nu))
#     lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
#
#     particles.mus[idx] = mu
#     particles.lambdas[idx] = lam


@wp.kernel
def k_particles_fill_deformations(particles: Particles):
    p = wp.tid()
    I = wp.identity(3, dtype=wp.float32)
    particles.F_Es[p] = I
    particles.F_Ps[p] = I
