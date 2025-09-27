"""Testing functions for incoherent sources.

We check that we are constructing the correct matrices for:
- non-orthogonal vector space linear algebra operations
- density operator
- derivative operator
- Lyapunov operator
- Quantum Fisher information

We test both the case of single point source distributions and the
vectorized version. We compare the results to values calculated in
a separate mathematica notebook ./non_orthogonal_transformation_matrix.nb.
"""

import numpy as np
import numpy.typing as npt

import rayfish.quantum_fisher_information.incoherent_sources as incoherent_sources
from rayfish import quantum_fisher_information as qfi


class TestGramian:
    """Test creation of the Gramian for non-orthogonal basis."""

    def test_delta_submat_scalar(
        self,
        mock_gramian_scalar: npt.NDArray[np.complex128],
        mock_coordinates_scalar: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_scalar: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing delta submatrix.

        Populating the delta submatrix with Gaussian overlaps
        obtained from gaussian_overlaps.delta(...). Calculation
        done for a scalar input for point source positions.
        """
        expected = mock_gramian_scalar[:2, :2]
        actual = incoherent_sources.delta_submat(
            coordinates=mock_coordinates_scalar,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_scalar,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )
        assert actual.shape == (1, 2, 2)
        assert np.allclose(actual, expected, atol=tolerance)

    def test_delta_submat_vector(
        self,
        mock_gramian_vector: npt.NDArray[np.complex128],
        mock_coordinates_vector: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_vector: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing delta submatrix.

        Populating the delta submatrix with Gaussian overlaps
        obtained from gaussian_overlaps.delta(...). Calculation
        done for a vector input for point source positions.
        """
        expected = mock_gramian_vector[:, :2, :2]
        actual = incoherent_sources.delta_submat(
            coordinates=mock_coordinates_vector,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_vector,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )
        assert actual.shape == (2, 2, 2)
        assert np.allclose(actual, expected, atol=tolerance)

    def test_gamma_submat_scalar(
        self,
        mock_gramian_scalar: npt.NDArray[np.complex128],
        mock_coordinates_scalar: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_scalar: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing gamma submatrix.

        Populating the gamma submatrix with overlaps of
        Gaussians and their derivatives obtained from
        - gaussian_overlaps.gamma_s(...) for derivative in x-dimension,
        - gaussian_overlaps.gamma_t(...) for derivative in y-dimension,
        - gaussian_overlaps.gamma_p(...) for derivative in z-dimension.

        Calculation done for a scalar input for point source positions.
        """
        expected = mock_gramian_scalar
        expected[:2, :2] = 0.0
        expected[2:8, 2:8] = 0.0
        actual = incoherent_sources.gamma_submat(
            coordinates=mock_coordinates_scalar,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_scalar,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        assert actual.shape == (1, 8, 8)
        assert np.allclose(actual, expected, atol=tolerance)

    def test_gamma_submat_vector(
        self,
        mock_gramian_vector: npt.NDArray[np.complex128],
        mock_coordinates_vector: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_vector: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing gamma submatrix.

        Populating the gamma submatrix with overlaps of
        Gaussians and their derivatives obtained from
        - gaussian_overlaps.gamma_s(...) for derivative in x-dimension,
        - gaussian_overlaps.gamma_t(...) for derivative in y-dimension,
        - gaussian_overlaps.gamma_p(...) for derivative in z-dimension.

        Calculation done for a vector input for point source positions.
        """
        expected = mock_gramian_vector
        expected[:, :2, :2] = 0.0
        expected[:, 2:8, 2:8] = 0.0
        actual = incoherent_sources.gamma_submat(
            coordinates=mock_coordinates_vector,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_vector,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )
        assert actual.shape == (2, 8, 8)
        assert np.allclose(actual, expected, atol=tolerance)

    def test_tau_submat_scalar(
        self,
        mock_gramian_scalar: npt.NDArray[np.complex128],
        mock_coordinates_scalar: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_scalar: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing tau submatrix.

        Populating the tau submatrix with overlaps between
        Gaussian derivatives obtained from
        - gaussian_overlaps.tau_ss(...) for x-dimension,
        - gaussian_overlaps.tau_st(...) for x- and y-dimension,
        - gaussian_overlaps.tau_sp(...) for x- and z-dimension,
        - gaussian_overlaps.tau_tt(...) for y-dimension,
        - gaussian_overlaps.tau_tp(...) for y- and z-dimension,
        - gaussian_overlaps.tau_pp(...) for z-dimension.


        Calculation done for a scalar input for point source positions.
        """
        expected = mock_gramian_scalar[2:, 2:]
        actual = qfi.tau_submat(
            coordinates=mock_coordinates_scalar,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_scalar,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )
        assert actual.shape == (1, 6, 6)
        assert np.allclose(actual, expected, atol=tolerance)

    def test_tau_submat_vector(
        self,
        mock_gramian_vector: npt.NDArray[np.complex128],
        mock_coordinates_vector: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_vector: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing tau submatrix.

        Populating the tau submatrix with overlaps between
        Gaussian derivatives obtained from
        - gaussian_overlaps.tau_ss(...) for x-dimension,
        - gaussian_overlaps.tau_st(...) for x- and y-dimension,
        - gaussian_overlaps.tau_sp(...) for x- and z-dimension,
        - gaussian_overlaps.tau_tt(...) for y-dimension,
        - gaussian_overlaps.tau_tp(...) for y- and z-dimension,
        - gaussian_overlaps.tau_pp(...) for z-dimension.

        Calculation done for a vector input for point source positions.
        """
        expected = mock_gramian_vector[:, 2:, 2:]
        actual = qfi.tau_submat(
            coordinates=mock_coordinates_vector,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_vector,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )
        assert actual.shape == (2, 6, 6)
        assert np.allclose(actual, expected, atol=tolerance)

    def test_gramian_scalar(
        self,
        mock_gramian_scalar: npt.NDArray[np.complex128],
        mock_coordinates_scalar: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_scalar: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing the Gramian.

        Populating the Gramian matrix with:
        - delta_submat
        - gamma_submat
        - tau_submat

        Calculation done for a scalar input for point source positions.
        """
        expected = mock_gramian_scalar

        delta_submat = incoherent_sources.delta_submat(
            coordinates=mock_coordinates_scalar,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_scalar,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        gamma_submat = incoherent_sources.gamma_submat(
            coordinates=mock_coordinates_scalar,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_scalar,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        tau_submat = qfi.tau_submat(
            coordinates=mock_coordinates_scalar,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_scalar,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        actual = incoherent_sources.gramian(
            delta_submat=delta_submat, gamma_submat=gamma_submat, tau_submat=tau_submat
        )

        assert actual.shape == (1, 8, 8)
        assert np.allclose(actual, expected, atol=tolerance)

    def test_gramian_vector(
        self,
        mock_gramian_vector: npt.NDArray[np.complex128],
        mock_coordinates_vector: npt.NDArray[np.float64],
        mock_number_of_point_sources: int,
        mock_number_of_point_source_distributions_vector: int,
        mock_wave_vector: int,
        mock_rayleigh_length: int,
        tolerance: float,
    ) -> None:
        """Test constructing the Gramian.

        Populating the Gramian matrix with:
        - delta_submat
        - gamma_submat
        - tau_submat

        Calculation done for a scalar input for point source positions.
        """
        expected = mock_gramian_vector

        delta_submat = incoherent_sources.delta_submat(
            coordinates=mock_coordinates_vector,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_vector,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        gamma_submat = incoherent_sources.gamma_submat(
            coordinates=mock_coordinates_vector,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_vector,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        tau_submat = qfi.tau_submat(
            coordinates=mock_coordinates_vector,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_vector,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        actual = incoherent_sources.gramian(
            delta_submat=delta_submat, gamma_submat=gamma_submat, tau_submat=tau_submat
        )

        assert actual.shape == (2, 8, 8)
        assert np.allclose(actual, expected, atol=tolerance)


def test_density(mock_density: npt.NDArray[np.float64]) -> None:
    """Test density matrix has correct shape and content."""
    expected = mock_density
    actual = incoherent_sources.density(
        number_of_point_sources=2,
    )

    assert actual.shape == (8, 8)
    assert np.trace(actual) == 1
    assert np.count_nonzero(actual) == 2
    assert np.array_equal(expected, actual)


def test_all_density_derivatives(
    mock_density_derivative: npt.NDArray[np.float64],
) -> None:
    """Test density derivatives matrices have correct shape and content."""
    expected = mock_density_derivative
    actual = incoherent_sources.all_density_derivatives(
        number_of_point_sources=2,
    )

    assert actual.shape == (6, 8, 8)
    for i in range(6):
        assert np.trace(actual[i]) == 0
        assert np.count_nonzero(actual[i]) == 2
    assert np.array_equal(expected, actual)


def test_pad_operators() -> None:
    """Test correct padding of rows and columns."""
    expected = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    mock_input = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    actual = incoherent_sources.quantum_fisher_information_matrix._pad_operator(
        matrix=mock_input, n=2, ind=2
    )
    assert np.array_equal(expected, actual)


def test_lyapunov_operators(
    mock_density: npt.NDArray[np.float64],
    mock_density_derivative: npt.NDArray[np.complex128],
    mock_gramian_scalar: npt.NDArray[np.complex128],
    mock_number_of_point_sources: int,
    mock_lyapunov_operators: npt.NDArray[np.complex128],
    tolerance: float,
) -> None:
    """Test Lyapunov operators.

    Testing the scipy.linalg.solve_silvester(...) yields accurate
    enough results and that the operators are padded correctly.
    """
    expected = mock_lyapunov_operators

    actual = incoherent_sources.lyapunov_operators(
        density=mock_density,
        all_density_derivatives=mock_density_derivative,
        gramian_of_single_distribution=mock_gramian_scalar,
        number_of_point_sources=mock_number_of_point_sources,
    )

    assert actual.shape == (6, 8, 8)
    # Adjusting tolerance due to numerics
    assert np.allclose(actual, expected, atol=10 * tolerance)


def test_fisher_information_matrix(
    mock_covariance_matrix: npt.NDArray[np.complex128],
    mock_density_derivative: npt.NDArray[np.complex128],
    mock_gramian_scalar: npt.NDArray[np.complex128],
    mock_lyapunov_operators: npt.NDArray[np.complex128],
    mock_jacobian_two_sources: npt.NDArray[np.float64],
    mock_number_of_point_sources: int,
    tolerance: float,
) -> None:
    """Test covariance matrix for Lyapunov operators.

    We test the correct construction of the matrix elements
    and re-parametrisation from absolute point source coordinates
    to relative coordinates change of basis.
    """
    expected = mock_covariance_matrix

    actual = incoherent_sources.fisher_information_matrix(
        all_density_derivatives=mock_density_derivative,
        gramian_of_single_distribution=mock_gramian_scalar,
        lyapunov_operators=mock_lyapunov_operators,
        coordinate_jacobian=mock_jacobian_two_sources,
        number_of_point_sources=mock_number_of_point_sources,
    )

    assert actual.shape == (3, 3)
    # Adjusting tolerance due to numerics
    assert np.allclose(actual, expected, atol=tolerance)
