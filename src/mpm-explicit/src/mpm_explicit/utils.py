
import warp as wp


@wp.func
def bspline_w(x: float) -> float:
    x = wp.abs(x)
    if x < 1.0:
        return 0.5 * x * x * x - x * x + 2.0 / 3.0
    elif x < 2.0:
        return -1.0 / 6.0 * x * x * x + x * x - 2.0 * x + 4.0 / 3.0
    return 0.0


@wp.func
def bspline_dw(x: float) -> float:
    x_abs = wp.abs(x)
    if x_abs < 1.0:
        return 1.5 * x_abs * x - 2.0 * x
    elif x_abs < 2.0:
        return -0.5 * x_abs * x + 2.0 * x - 2.0 * wp.sign(x)
    return 0.0


@wp.func
def safe_svd3(M: wp.mat33):
    U = wp.mat33(1.0)
    V = wp.mat33(1.0)
    sigma = wp.vec3(0.0)

    wp.svd3(M, U, sigma, V)

    if wp.determinant(U) < 0.0:
        U = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2],
        )
        sigma[2] = -sigma[2]

    if wp.determinant(V) < 0.0:
        V = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2],
        )
        sigma[2] = -sigma[2]

    return U, sigma, V


@wp.func
def extract_rotation(F: wp.mat33) -> wp.mat33:
    U, _sigma, V = safe_svd3(F)

    return U * wp.transpose(V)


@wp.func
def cofactor(A: wp.mat33) -> wp.mat33:
    c00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    c01 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    c02 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]

    c10 = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    c11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    c12 = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]

    c20 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    c21 = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    c22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    return wp.mat33(
        c00, c01, c02,
        c10, c11, c12,
        c20, c21, c22
    )
