import warp as wp

DEFAULT_DENSITY = wp.constant(200.0)
DEFAULT_E = wp.constant(1.4e5)
DEFAULT_NU =  wp.constant(0.2)
DEFAULT_THETA_C = wp.constant(2.5e-2)
DEFAULT_THETA_S = wp.constant(7.5e-3)
DEFAULT_XI = wp.constant(10)

EPSILON = wp.constant(1e-8)
EPSILON_SQ = wp.constant(1e-9)
GRAVITY = wp.constant(wp.vec3(0.0, 0.0, -4.9))
COULOMB_FRICTION = wp.constant(0.9)
PICFLIP_ALPHA = wp.constant(0.95)
MAX_COLLISION_DIST = wp.constant(0.02)

# Adaptive (CFL-based) timestep, recomputed every step from the particles'
# current max speed: dt = min(CFL * min_cell_size / sqrt(max_speed^2), MAX_TIMESTEP).
CFL = wp.constant(0.04)
MAX_TIMESTEP = wp.constant(5e-4)
MIN_TIMESTEP = wp.constant(1e-6)

# Density-based particle shading (grayscale):
# color = clamp(relative_density * contrast + (1 - contrast), 0, 1), so
# particles at their reference density render white/bright and particles
# that have fluffed up/fractured (locally less dense) render darker.
RENDER_DENSITY_CONTRAST = 0.5
