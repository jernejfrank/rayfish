"""Tests for the parameter-catalogue Jacobians (restores deleted coverage)."""

import numpy as np
import pytest

from rayfish.parameter_catalogue.distance_2_sources_x import (
    coordinate_jacobian as jacobian_two_sources,
)
from rayfish.parameter_catalogue.distance_3_sources_xz import (
    coordinate_jacobian__no_intensity,
    coordinate_jacobian__with_intensity,
)


def test_two_source_jacobian_layout() -> None:
    """Separation Jacobian: sources at -/+ d/2 along x, 8 columns (3N + N)."""
    expected = np.array([[-1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(jacobian_two_sources(), expected)


def test_three_source_jacobian_no_intensity() -> None:
    """Width/depth Jacobian: dx := dx_3 - dx_2, dz := -dz_1, intensity columns zero."""
    expected = np.array([
        [0, -1 / 2, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    ])
    np.testing.assert_array_equal(coordinate_jacobian__no_intensity(), expected)


@pytest.mark.parametrize("target_idx", [0, 1, 2])
def test_three_source_jacobian_with_intensity(target_idx: int) -> None:
    """Intensity Jacobian row is trace-preserving and works for every target source.

    Includes target_idx = 2 (the last source), which used to raise an IndexError due
    to a vacuously-true bounds check when looking up a neighbouring intensity.
    """
    intensities = [1 / 3, 1 / 3, 1 / 3]
    jacobian = coordinate_jacobian__with_intensity(
        point_source_intensities=intensities,
        target_point_source_idx=target_idx,
    )

    assert jacobian.shape == (3, 12)
    # Spatial rows are unchanged from the no-intensity catalogue.
    np.testing.assert_array_equal(jacobian[:2], coordinate_jacobian__no_intensity())

    intensity_row = jacobian[2, 9:]
    # dI's sum to zero: the reparametrization preserves the total photon budget.
    np.testing.assert_allclose(intensity_row.sum(), 0.0, atol=1e-15)
    # Equal intensities: (V-1) * I^2 on the target, -I^2 elsewhere.
    expected_row = np.full(3, -1 / 9)
    expected_row[target_idx] = 2 / 9
    np.testing.assert_allclose(intensity_row, expected_row, atol=1e-15)
