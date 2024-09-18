"""Some constants for general coordinate transformations."""

import numpy as np
import numpy.typing as npt
import pytest


# Due to numerical precision we cannot get exact equality
@pytest.fixture()
def tolerance() -> float:
    """Tolerance for asserting two values are the same."""
    return 5e-6


@pytest.fixture
def mock_wave_vector() -> int:
    """Good value for exponential."""
    return 2


@pytest.fixture
def mock_rayleigh_length() -> int:
    """Good value for exponential."""
    return 1


@pytest.fixture
def mock_number_of_point_sources() -> int:
    """Replicating Tsang's original setup."""
    return 2


@pytest.fixture
def mock_number_of_point_source_distributions_scalar() -> int:
    """For testing single case."""
    return 1


@pytest.fixture
def mock_coordinates_scalar() -> npt.NDArray[np.float64]:
    """For testing single case."""
    return np.array([[1.3, 0.7, 0.93, 1.17, 0.23, 0.0654]])


@pytest.fixture
def mock_number_of_point_source_distributions_vector() -> int:
    """For testing multiple cases as vector."""
    return 2


@pytest.fixture
def mock_coordinates_vector() -> npt.NDArray[np.float64]:
    """For testing multiple cases as vector."""
    return np.array([
        [1.3, 0.7, 0.93, 1.17, 0.23, 0.0654],
        [1.1, -1.1, 0.04, -0.04, 0.0054, -0.0054],
    ])


@pytest.fixture
def mock_jacobian_two_sources() -> npt.NDArray[np.float64]:
    """Jacobian for two point sources."""
    return np.array([
        [-1 / 2, 1 / 2, 0, 0, 0, 0],
        [0, 0, -1 / 2, 1 / 2, 0, 0],
        [0, 0, 0, 0, -1 / 2, 1 / 2],
    ])


@pytest.fixture
def mock_spacing_two_sources() -> npt.NDArray[np.float64]:
    """Spacing for two point sources."""
    return np.array([0.2, 0.4, 0.6])


@pytest.fixture
def mock_spacing_three_sources() -> npt.NDArray[np.float64]:
    """Spacing for three point sources."""
    return np.array([3, 6, 9, 12])
