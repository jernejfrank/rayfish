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

import rayfish.quantum_fisher_information as incoherent_sources
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
        actual = qfi.delta_submat(
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
        actual = qfi.delta_submat(
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
        actual = qfi.gamma_submat(
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
        actual = qfi.gamma_submat(
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

        delta_submat = qfi.delta_submat(
            coordinates=mock_coordinates_scalar,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_scalar,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        gamma_submat = qfi.gamma_submat(
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

        delta_submat = qfi.delta_submat(
            coordinates=mock_coordinates_vector,
            number_of_point_sources=mock_number_of_point_sources,
            number_of_point_source_distributions=mock_number_of_point_source_distributions_vector,
            wave_vector=mock_wave_vector,
            rayleigh_length=mock_rayleigh_length,
        )

        gamma_submat = qfi.gamma_submat(
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
    """Test density matrix has correct shape and content.

    Implements R_Δ = diag(I_1, ..., I_V) embedded in the (4N,4N) block matrix,
    i.e. the density matrix R = Σ_v I_v |ψ_v><ψ_v| of Eq. (eq:incoherent_ensemble)
    in the non-orthogonal basis. Equal intensities I_v = 1/V recover Tr R = 1.
    """
    expected = mock_density
    actual = incoherent_sources.density(
        number_of_point_sources=2,
        point_source_intensities=[0.5, 0.5],
    )

    assert actual.shape == (8, 8)
    assert np.trace(actual) == 1
    assert np.count_nonzero(actual) == 2
    assert np.array_equal(expected, actual)


def test_density_derivatives(
    mock_density_derivative: npt.NDArray[np.float64],
) -> None:
    """Test density derivatives matrices have correct shape and content.

    Implements the sparse derivative matrices D_a with the two non-zero entries
    D_a^{∂u,u} = D_a^{u,∂u} = I_u (Eq. between eatlean and eq:qfim_final). Each
    derivative is traceless and has exactly two non-zero entries.
    """
    expected = mock_density_derivative
    actual = incoherent_sources.density_derivatives(
        number_of_point_sources=2,
        point_source_intensities=[0.5, 0.5],
        include_intensity_params=False,
    )

    assert actual.shape == (6, 8, 8)
    for i in range(6):
        assert np.trace(actual[i]) == 0
        assert np.count_nonzero(actual[i]) == 2
    assert np.array_equal(expected, actual)


def test_include_intensity_params() -> None:
    """The intensity flag is inferred from the coordinate Jacobian width.

    A width of 3N re-parametrizes position only; the extra N columns (width 4N)
    carry the per-source intensity parameters.
    """
    number_of_point_sources = 2
    position_only_jacobian = np.zeros((5, 3 * number_of_point_sources))
    with_intensity_jacobian = np.zeros((5, 4 * number_of_point_sources))

    assert not incoherent_sources.include_intensity_params(
        position_only_jacobian, number_of_point_sources
    )
    assert incoherent_sources.include_intensity_params(
        with_intensity_jacobian, number_of_point_sources
    )


def test_density_derivatives_include_intensity(
    mock_density_derivative: npt.NDArray[np.float64],
) -> None:
    """Treating intensity as a parameter appends N diagonal intensity derivatives.

    With ``include_intensity_params`` the first axis grows from 3N to 4N. The 3N
    spatial derivatives are unchanged, and each extra matrix is the diagonal indicator
    D_a^{u,u} = 1 (trace 1, a single non-zero entry).
    """
    actual = incoherent_sources.density_derivatives(
        number_of_point_sources=2,
        point_source_intensities=[0.5, 0.5],
        include_intensity_params=True,
    )

    assert actual.shape == (8, 8, 8)
    assert np.array_equal(actual[:6], mock_density_derivative)
    for source in range(2):
        intensity_derivative = actual[6 + source]
        assert np.trace(intensity_derivative) == 1
        assert np.count_nonzero(intensity_derivative) == 1
        assert intensity_derivative[source, source] == 1


def test_lyapunov_operators(
    mock_density: npt.NDArray[np.float64],
    mock_density_derivative: npt.NDArray[np.complex128],
    mock_gramian_scalar: npt.NDArray[np.complex128],
    mock_number_of_point_sources: int,
    tolerance: float,
) -> None:
    """Test Lyapunov operators solve the reduced Sylvester equation.

    The reduced density block diag(I_1, ..., I_V, 0) is singular, so the SLD
    Eq. (eq:sld_nonorthogonal) 2 D_a = R G L_a + L_a G R has a non-unique
    solution; the implementation returns the minimum-norm one via a pinv on the
    Kronecker-vectorized system. We therefore test the *defining property* on
    each (N+1)-dimensional subsystem,

        (R G) X + X (G R) = 2 D_a,

    rather than exact operator values, plus the correct sparse padding back to
    the full (4N,4N) shape.
    """
    n_sources = mock_number_of_point_sources
    n_dist = 1
    gramian = mock_gramian_scalar[None]

    actual = incoherent_sources.lyapunov_operators(
        density=mock_density,
        density_derivatives=mock_density_derivative,
        gramian=gramian,
        number_of_point_sources=n_sources,
        number_of_point_source_distributions=n_dist,
    )

    assert actual.shape == (n_dist, 3 * n_sources, 8, 8)

    # Each operator must satisfy the reduced Sylvester equation on its subsystem.
    for i in range(3 * n_sources):
        _slice = np.arange(0, n_sources + 1)
        _slice[-1] = n_sources + i
        sub = actual[0, i][np.ix_(_slice, _slice)]
        density_sub = mock_density[np.ix_(_slice, _slice)]
        gramian_sub = mock_gramian_scalar[np.ix_(_slice, _slice)]
        a = density_sub @ gramian_sub
        b = gramian_sub @ density_sub
        rhs = (2 * mock_density_derivative[i])[np.ix_(_slice, _slice)]
        residual = a @ sub + sub @ b - rhs
        assert np.allclose(residual, 0.0, atol=10 * tolerance)

        # Rows/columns outside the subsystem must remain zero (sparse padding).
        mask = np.ones(8, dtype=bool)
        mask[_slice] = False
        assert np.allclose(actual[0, i][mask, :], 0.0)
        assert np.allclose(actual[0, i][:, mask], 0.0)


def test_lyapunov_operators_include_intensity(
    mock_density: npt.NDArray[np.float64],
    mock_density_derivative: npt.NDArray[np.complex128],
    mock_gramian_scalar: npt.NDArray[np.complex128],
    mock_number_of_point_sources: int,
    tolerance: float,
) -> None:
    """With intensity parameters, the trailing N operators solve the source block.

    When the derivative tensor carries the intensity block (first axis 4N), each of the
    N intensity operators solves the reduced Sylvester equation

        (R G) X + X (G R) = 2 D_a,

    restricted to the N-dimensional source populations (no tangent index), and is padded
    with zeros back to the full (4N,4N) shape.
    """
    n_sources = mock_number_of_point_sources
    n_dist = 1
    gramian = mock_gramian_scalar[None]

    # Extend the position-only derivatives with the N diagonal intensity derivatives.
    intensity_blocks = np.zeros((n_sources, 8, 8), dtype=np.complex128)
    for source in range(n_sources):
        intensity_blocks[source, source, source] = 1
    density_derivatives = np.concatenate([mock_density_derivative, intensity_blocks])

    actual = incoherent_sources.lyapunov_operators(
        density=mock_density,
        density_derivatives=density_derivatives,
        gramian=gramian,
        number_of_point_sources=n_sources,
        number_of_point_source_distributions=n_dist,
    )

    assert actual.shape == (n_dist, 4 * n_sources, 8, 8)

    _slice = np.arange(n_sources)
    density_sub = mock_density[np.ix_(_slice, _slice)]
    gramian_sub = mock_gramian_scalar[np.ix_(_slice, _slice)]
    a = density_sub @ gramian_sub
    b = gramian_sub @ density_sub
    for i in range(n_sources):
        operator = actual[0, 3 * n_sources + i]
        sub = operator[np.ix_(_slice, _slice)]
        rhs = (2 * density_derivatives[3 * n_sources + i])[np.ix_(_slice, _slice)]
        residual = a @ sub + sub @ b - rhs
        assert np.allclose(residual, 0.0, atol=10 * tolerance)

        # Only the source block is populated for intensity operators.
        mask = np.ones(8, dtype=bool)
        mask[_slice] = False
        assert np.allclose(operator[mask, :], 0.0)
        assert np.allclose(operator[:, mask], 0.0)


def test_fisher_information_matrix(
    mock_covariance_matrix: npt.NDArray[np.complex128],
    mock_density_derivative: npt.NDArray[np.complex128],
    mock_gramian_scalar: npt.NDArray[np.complex128],
    mock_lyapunov_operators: npt.NDArray[np.complex128],
    mock_jacobian_two_sources: npt.NDArray[np.float64],
    mock_number_of_point_sources: int,
    tolerance: float,
) -> None:
    """Test the quantum Fisher information matrix construction.

    Implements H_{a,b} = Re Tr(L_a G D_b G) (Eq. eq:qfi_nonorthogonal) followed
    by the Jacobian change of basis H̃ = B H B^T (Eq. def:jacobian). The QFIM is
    invariant to the non-uniqueness of the SLD, so we feed the reference Lyapunov
    operators and check the matrix elements and the re-parametrisation from
    absolute point source coordinates to the parameters of interest.
    """
    expected = mock_covariance_matrix
    n_dist = 1

    actual = incoherent_sources.fisher_information_matrix_for_entire_distribution(
        density_derivatives=mock_density_derivative,
        gramian=mock_gramian_scalar[None],
        lyapunov_operators=mock_lyapunov_operators[None],
        coordinate_jacobian=mock_jacobian_two_sources,
        number_of_point_sources=mock_number_of_point_sources,
        number_of_point_source_distributions=n_dist,
    )

    assert actual.shape == (n_dist, 3, 3)
    # Adjusting tolerance due to numerics
    assert np.allclose(actual[0], expected, atol=tolerance)
