"""Testing functions for incoherent sources.

We check that we are constructing the correct matrices for:
- non-orthogonal vector space linear algebra operations
- density operator
- derivative operator
- Lyapunov operator
- Quantum Fisher information

We test both the case of single point source distributions and the
vectorized version.
"""

# import numpy as np

# WAVE_VECTOR = 2
# RAYLEIGH_LENGTH = 1
# TESTSTING_COORDINATES_SCALAR = np.array([1.3, 0.7, 0.93, 1.17, 0.23, 0.0654])
# TESTSTING_COORDINATES_VECTOR = np.array([
#     [1.3, 0.7, 0.93, 1.17, 0.23, 0.0654],
#     [1.1, -1.1, 0.04, -0.04, 0.0054, -0.0054],
# ])

# G_SCALAR = np.array([
#     [1, 0.869266 + 0.227171j, 0, 0, -((3j) / 2), 0.264597 + 0.046375j],
#     [
#         0.869266 - 0.227171j,
#         1,
#         -0.264597 + 0.046375j,
#         0.105839 - 0.01855j,
#         -0.381302 - 1.34393j,
#         0,
#     ],
#     [0, -0.264597 - 0.046375j, 1 / 2, 0, 0, 0.361012 + 0.0699617j],
#     [0, 0.105839 + 0.01855j, 0, 1 / 2, 0, 0.0319929 + 0.00293198j],
#     [(3j) / 2, -0.381302 + 1.34393j, 0, 0, 5 / 2, -0.0684459 + 0.276513j],
#     [
#         0.264597 - 0.046375j,
#         0,
#         0.361012 - 0.0699617j,
#         0.0319929 - 0.00293198j,
#         -0.0684459 - 0.276513j,
#         1 / 2,
#     ],
# ])

# # Due to numerical precision we cannot get exact equality
# TOLERANCE = 5e-6


# def test_delta_submat_scalar() -> None:
#     expected = G_SCALAR[:2, :2]
#     # actual = qfi.delta_submat(coordinates=TESTSTING_COORDINATES_SCALAR)


# if __name__ == "__main__":
#     print(TESTSTING_COORDINATES)
#     print(G_SCALAR)
