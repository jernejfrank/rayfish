"""Testing the mode-mixing helpers for imperfect mode sorting.

The Hermite-Gaussian construction itself is taken as given. Here we verify the two
private helpers that model an imperfect (cross-talking) mode sorter:

- ``_mixing_unitary`` builds a nearest-neighbour Givens rotation on the mode index.
- ``_mix_modes_via_unitary`` applies that rotation to the per-axis mode vectors and
  rebuilds the flattened product basis.

We exercise dimension 2 (the baseline single-Givens case) and dimension 3 (the first
case where the chained rotations interact), comparing against reference matrices that
are derived by hand rather than re-implementing the function under test.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from rayfish.classical_fisher_information.measurement_basis.hermite_gaussian_basis import (  # noqa: E501
    _mix_modes_via_unitary,
    _mixing_unitary,
    measurement_basis,
)


def _reference_unitary(dim: int, epsilon: float) -> npt.NDArray[np.float64]:
    """Hand-derived nearest-neighbour Givens chain ``G_{d-2} @ ... @ G_1 @ G_0``.

    Closed forms obtained independently of the implementation: for ``dim == 2`` this is
    a single plane rotation, and for ``dim == 3`` it is the product of the (0,1) and
    (1,2) rotations, where the middle order is rotated twice.
    """
    c, s = np.cos(epsilon), np.sin(epsilon)
    if dim == 2:
        return np.array([[c, -s], [s, c]])
    if dim == 3:
        return np.array([
            [c, -s, 0.0],
            [c * s, c**2, -s],
            [s**2, s * c, c],
        ])
    raise ValueError(f"No hand-derived reference for dim={dim}.")


def _identity_basis(dim: int) -> list[dict[str, npt.NDArray[np.float64]]]:
    """Synthetic product basis whose per-axis vectors are the standard basis.

    Using ``e_a`` for the x-mode and ``e_b`` for the y-mode lets us read off the applied
    unitary directly, isolating the mixer's extraction/reorder logic from the Hermite
    coefficients.
    """
    eye = np.eye(dim)
    return [
        {"x": eye[a].copy(), "y": eye[b].copy()} for a in range(dim) for b in range(dim)
    ]


# --------------------------------------------------------------------------------------
# _mixing_unitary
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("epsilon", [0.05, 0.3, 1.0])
def test_mixing_unitary_is_orthogonal(dim: int, epsilon: float) -> None:
    """The rotation is orthogonal for any angle, so mixed modes stay orthonormal."""
    unitary = _mixing_unitary(dim, epsilon)
    assert np.allclose(unitary @ unitary.T, np.eye(dim))


@pytest.mark.parametrize("dim", [2, 3])
def test_mixing_unitary_identity_at_zero(dim: int) -> None:
    """Zero mixing strength must reduce to the identity (perfect sorter)."""
    assert np.allclose(_mixing_unitary(dim, 0.0), np.eye(dim))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("epsilon", [0.3, 1.0])
def test_mixing_unitary_is_proper_rotation(dim: int, epsilon: float) -> None:
    """A product of Givens rotations has unit determinant (no reflection)."""
    assert np.isclose(np.linalg.det(_mixing_unitary(dim, epsilon)), 1.0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("epsilon", [0.05, 0.3, 1.0])
def test_mixing_unitary_matches_reference(dim: int, epsilon: float) -> None:
    """The chained Givens rotation matches the hand-derived closed form."""
    actual = _mixing_unitary(dim, epsilon)
    expected = _reference_unitary(dim, epsilon)
    assert np.allclose(actual, expected)


def test_mixing_unitary_only_couples_nearest_neighbours_to_first_order() -> None:
    """For small angles, leakage is nearest-neighbour: |n><n+-1| entries are O(eps).

    The asymmetric (2,0) entry is O(eps^2) (the middle order is rotated twice), so it
    stays negligible relative to the nearest-neighbour entries, while the (0,2) entry is
    exactly zero.
    """
    epsilon = 1e-3
    unitary = _mixing_unitary(3, epsilon)
    assert np.isclose(unitary[0, 1], -epsilon, atol=1e-6)
    assert np.isclose(unitary[1, 0], epsilon, atol=1e-6)
    assert unitary[0, 2] == 0.0
    assert abs(unitary[2, 0]) < epsilon  # O(eps^2) double-rotation leakage


# --------------------------------------------------------------------------------------
# _mix_modes_via_unitary
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_mix_modes_zero_strength_reproduces_basis(dim: int) -> None:
    """Zero mixing leaves the Hermite basis untouched (regression safety net)."""
    base = measurement_basis(dim)
    mixed = _mix_modes_via_unitary(base, 0.0)
    assert len(mixed) == len(base)
    for original, new in zip(base, mixed):
        assert np.array_equal(original["x"], new["x"])
        assert np.array_equal(original["y"], new["y"])


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("epsilon", [0.2, 0.7])
def test_mix_modes_applies_unitary_per_axis(dim: int, epsilon: float) -> None:
    """Each axis is mixed by the unitary: mode (a,b) becomes (U[a], U[b]).

    Feeding the standard-basis vectors makes ``U @ I == U``, so the mixed x-vector of
    channel ``a*dim + b`` must equal row ``a`` of ``U`` and the y-vector row ``b``. This
    pins down extraction stride, per-axis application, and the product-order rebuild.
    """
    unitary = _mixing_unitary(dim, epsilon)
    mixed = _mix_modes_via_unitary(_identity_basis(dim), epsilon)
    for a in range(dim):
        for b in range(dim):
            channel = mixed[a * dim + b]
            assert np.allclose(channel["x"], unitary[a])
            assert np.allclose(channel["y"], unitary[b])


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("epsilon", [0.2, 0.7])
def test_mix_modes_x_depends_only_on_outer_index(dim: int, epsilon: float) -> None:
    """The flattened order is preserved: x-mode is the outer index, y-mode the inner."""
    mixed = _mix_modes_via_unitary(measurement_basis(dim), epsilon)
    for a in range(dim):
        x_reference = mixed[a * dim + 0]["x"]
        for b in range(dim):
            assert np.allclose(mixed[a * dim + b]["x"], x_reference)
    for b in range(dim):
        y_reference = mixed[0 * dim + b]["y"]
        for a in range(dim):
            assert np.allclose(mixed[a * dim + b]["y"], y_reference)


def test_mix_modes_blends_first_order_neighbours() -> None:
    """A mixed Hermite mode is its own order plus an O(eps) slice of its neighbours.

    For dim 3, the x-mode of order 1 should become eps*H0 + H1 - eps*H2 to first order,
    the physical nearest-neighbour cross-talk we are modelling.
    """
    dim, epsilon = 3, 1e-4
    base = measurement_basis(dim)
    hermite_x = [base[a * dim]["x"] for a in range(dim)]
    mixed = _mix_modes_via_unitary(base, epsilon)
    phi_1 = mixed[1 * dim + 0]["x"]
    expected = epsilon * hermite_x[0] + hermite_x[1] - epsilon * hermite_x[2]
    assert np.allclose(phi_1, expected, atol=1e-7)
