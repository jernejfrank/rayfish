"""Imperfect mode-sorting study for the 3-point-source x/z geometry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from rayfish.classical_fisher_information.incoherent import (
    amplitude_measurement_model as amplitude_model,
)
from rayfish.classical_fisher_information.incoherent import (
    covariance_matrix_for_entire_distribution,
    imperfect_mode_sorting,
)
from rayfish.classical_fisher_information.incoherent import (
    fisher_information_matrix as ideal_fisher_information,
)
from rayfish.classical_fisher_information.measurement_basis import (
    hermite_gaussian_basis_x_only,
)
from rayfish.parameter_catalogue.distance_3_sources_xz import coordinate_jacobian

OUTPUT_DIRECTORY = Path(__file__).parent
WAVE_VECTOR = 2.0
RAYLEIGH_LENGTH = 1.0
NUMBER_OF_POINT_SOURCES = 3
MEASURED_MODES = 6
PADDED_MODES = 10
TARGET_PARAMETER_INDEX = 1
REPRESENTATIVE_Z = 0.08
X_SEPARATION = 0.12
EPSILONS = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
DEFECT_DISTANCES = np.linspace(0.01, 0.2, 25)
N_REALIZATIONS = 32
PLOT_EPSILONS = (0.0, 0.1, 0.2)
PLOT_COLORS = ("#0f172a", "#0f766e", "#b45309")


@dataclass(frozen=True)
class RegimeResult:
    """Summary arrays for one imperfection regime."""

    fi_ratio_mean: np.ndarray
    fi_ratio_std: np.ndarray
    crb_mean: np.ndarray
    crb_std: np.ndarray


def symmetric_three_source_coordinates(
    x_separation: float, z_displacement: float
) -> np.ndarray:
    """Three-source x/z geometry used in the existing example family."""
    return np.array([
        [0.0, -x_separation, x_separation, 0.0, 0.0, 0.0, -z_displacement, 0.0, 0.0]
    ])


def random_orthogonal_matrix(
    size: int, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a seeded orthogonal matrix via a skew-symmetric exponential."""
    generator = rng.normal(size=(size, size))
    generator = generator - generator.T
    return expm(epsilon * generator)


def lossy_truncation_matrix(
    measured_modes: int,
    padded_modes: int,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Project a larger unitary back onto the retained basis to induce loss."""
    full_unitary = random_orthogonal_matrix(padded_modes, epsilon, rng)
    return full_unitary[:measured_modes, :measured_modes]


def ideal_probability_and_fi(
    coordinates: np.ndarray, measurement_basis: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ideal per-mode FI, total FI, and covariance."""
    probability = ideal_fisher_information.probability__1D(
        coordinates=coordinates,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
        number_of_point_source_distributions=1,
        measurement_basis=measurement_basis,
        wave_vector=WAVE_VECTOR,
        rayleigh_length=RAYLEIGH_LENGTH,
    )
    fi_per_mode = (
        ideal_fisher_information.fisher_information_matrix_for_entire_distribution(
            coordinates=coordinates,
            number_of_point_sources=NUMBER_OF_POINT_SOURCES,
            number_of_point_source_distributions=1,
            measurement_basis=measurement_basis,
            probability=probability,
            wave_vector=WAVE_VECTOR,
            rayleigh_length=RAYLEIGH_LENGTH,
            coordinate_jacobian=coordinate_jacobian(),
        )
    )
    total_fi = np.sum(fi_per_mode, axis=1)
    covariance = covariance_matrix_for_entire_distribution(fi_per_mode)
    return probability, total_fi[0], covariance[0]


def imperfect_probability_and_fi(
    coordinates: np.ndarray,
    measurement_basis: list[dict[str, np.ndarray]],
    mode_mixing_matrix: np.ndarray,
    include_loss_channel: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the imperfect per-mode FI, total FI, and covariance."""
    probability = imperfect_mode_sorting.probability__1D(
        coordinates=coordinates,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
        number_of_point_source_distributions=1,
        measurement_basis=measurement_basis,
        mode_mixing_matrix=mode_mixing_matrix,
        include_loss_channel=include_loss_channel,
        wave_vector=WAVE_VECTOR,
        rayleigh_length=RAYLEIGH_LENGTH,
    )
    fi_per_mode = (
        imperfect_mode_sorting.fisher_information_matrix_for_entire_distribution(
            coordinates=coordinates,
            number_of_point_sources=NUMBER_OF_POINT_SOURCES,
            number_of_point_source_distributions=1,
            measurement_basis=measurement_basis,
            probability=probability,
            wave_vector=WAVE_VECTOR,
            rayleigh_length=RAYLEIGH_LENGTH,
            coordinate_jacobian=coordinate_jacobian(),
            mode_mixing_matrix=mode_mixing_matrix,
            include_loss_channel=include_loss_channel,
        )
    )
    total_fi = np.sum(fi_per_mode, axis=1)
    covariance = covariance_matrix_for_entire_distribution(fi_per_mode)
    return probability, total_fi[0], covariance[0]


def truncation_loss_probability_and_fi(
    coordinates: np.ndarray,
    measured_basis: list[dict[str, np.ndarray]],
    padded_basis: list[dict[str, np.ndarray]],
    mode_mixing_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model loss by mixing in a larger basis and retaining only the measured modes."""
    amplitudes = amplitude_model.mode_amplitudes(
        coordinates=coordinates,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
        measurement_basis=padded_basis,
        wave_vector=WAVE_VECTOR,
        rayleigh_length=RAYLEIGH_LENGTH,
    )
    derivatives_s, derivatives_t, derivatives_p = (
        amplitude_model.mode_amplitude_derivatives(
            coordinates=coordinates,
            number_of_point_sources=NUMBER_OF_POINT_SOURCES,
            measurement_basis=padded_basis,
            wave_vector=WAVE_VECTOR,
            rayleigh_length=RAYLEIGH_LENGTH,
        )
    )

    mixed_amplitudes = np.einsum("ab,mbs->mas", mode_mixing_matrix, amplitudes)
    mixed_derivatives_s = np.einsum("ab,mbs->mas", mode_mixing_matrix, derivatives_s)
    mixed_derivatives_t = np.einsum("ab,mbs->mas", mode_mixing_matrix, derivatives_t)
    mixed_derivatives_p = np.einsum("ab,mbs->mas", mode_mixing_matrix, derivatives_p)

    retained_amplitudes = mixed_amplitudes[:, : len(measured_basis), :]
    retained_derivatives_s = mixed_derivatives_s[:, : len(measured_basis), :]
    retained_derivatives_t = mixed_derivatives_t[:, : len(measured_basis), :]
    retained_derivatives_p = mixed_derivatives_p[:, : len(measured_basis), :]

    probability = amplitude_model.probabilities_from_amplitudes(
        amplitudes=retained_amplitudes,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
    )
    probability_derivatives = amplitude_model.probability_derivatives_from_amplitudes(
        amplitudes=retained_amplitudes,
        derivatives_s=retained_derivatives_s,
        derivatives_t=retained_derivatives_t,
        derivatives_p=retained_derivatives_p,
        number_of_point_sources=NUMBER_OF_POINT_SOURCES,
    )
    (
        probability,
        derivatives_s_probability,
        derivatives_t_probability,
        derivatives_p_probability,
    ) = imperfect_mode_sorting.append_loss_channel(
        probability=probability,
        derivatives_s=probability_derivatives[0],
        derivatives_t=probability_derivatives[1],
        derivatives_p=probability_derivatives[2],
    )

    fi_per_mode = amplitude_model.projected_fisher_information_per_mode(
        probability=probability,
        derivatives_s=derivatives_s_probability,
        derivatives_t=derivatives_t_probability,
        derivatives_p=derivatives_p_probability,
        coordinate_jacobian=coordinate_jacobian(),
    )
    total_fi = np.sum(fi_per_mode, axis=1)
    covariance = covariance_matrix_for_entire_distribution(fi_per_mode)
    return probability, total_fi[0], covariance[0]


def aggregate_regime(
    *,
    epsilons: np.ndarray,
    defect_distances: np.ndarray,
    n_realizations: int,
    measured_basis: list[dict[str, np.ndarray]],
    mixing_factory: Callable[[float, np.random.Generator], np.ndarray],
    lossy: bool,
) -> RegimeResult:
    """Aggregate FI ratios and CRBs over mismatch realizations."""
    representative_coordinates = symmetric_three_source_coordinates(
        x_separation=X_SEPARATION,
        z_displacement=REPRESENTATIVE_Z,
    )
    _, ideal_fi_reference, _ = ideal_probability_and_fi(
        coordinates=representative_coordinates,
        measurement_basis=measured_basis,
    )
    ideal_scalar_reference = ideal_fi_reference[
        TARGET_PARAMETER_INDEX, TARGET_PARAMETER_INDEX
    ]

    fi_ratio_mean = np.zeros_like(epsilons)
    fi_ratio_std = np.zeros_like(epsilons)
    crb_mean = np.zeros((len(epsilons), len(defect_distances)))
    crb_std = np.zeros((len(epsilons), len(defect_distances)))

    padded_basis = hermite_gaussian_basis_x_only.measurement_basis(PADDED_MODES)

    for epsilon_index, epsilon in enumerate(epsilons):
        fi_ratios = np.zeros(n_realizations)
        crb_curves = np.zeros((n_realizations, len(defect_distances)))
        for realization in range(n_realizations):
            rng = np.random.default_rng(seed=10_000 * epsilon_index + realization)
            mixing = mixing_factory(epsilon, rng)

            if lossy:
                _, imperfect_fi, _ = truncation_loss_probability_and_fi(
                    coordinates=representative_coordinates,
                    measured_basis=measured_basis,
                    padded_basis=padded_basis,
                    mode_mixing_matrix=mixing,
                )
            else:
                _, imperfect_fi, _ = imperfect_probability_and_fi(
                    coordinates=representative_coordinates,
                    measurement_basis=measured_basis,
                    mode_mixing_matrix=mixing,
                    include_loss_channel=False,
                )
            fi_ratios[realization] = (
                imperfect_fi[TARGET_PARAMETER_INDEX, TARGET_PARAMETER_INDEX]
                / ideal_scalar_reference
            )

            for defect_index, defect_distance in enumerate(defect_distances):
                coordinates = symmetric_three_source_coordinates(
                    x_separation=X_SEPARATION,
                    z_displacement=defect_distance,
                )
                if lossy:
                    _, _, covariance = truncation_loss_probability_and_fi(
                        coordinates=coordinates,
                        measured_basis=measured_basis,
                        padded_basis=padded_basis,
                        mode_mixing_matrix=mixing,
                    )
                else:
                    _, _, covariance = imperfect_probability_and_fi(
                        coordinates=coordinates,
                        measurement_basis=measured_basis,
                        mode_mixing_matrix=mixing,
                        include_loss_channel=False,
                    )
                crb_curves[realization, defect_index] = covariance[
                    TARGET_PARAMETER_INDEX, TARGET_PARAMETER_INDEX
                ]

        fi_ratio_mean[epsilon_index] = fi_ratios.mean()
        fi_ratio_std[epsilon_index] = fi_ratios.std(ddof=0)
        crb_mean[epsilon_index] = crb_curves.mean(axis=0)
        crb_std[epsilon_index] = crb_curves.std(axis=0, ddof=0)

    return RegimeResult(
        fi_ratio_mean=fi_ratio_mean,
        fi_ratio_std=fi_ratio_std,
        crb_mean=crb_mean,
        crb_std=crb_std,
    )


def plot_fi_ratio(
    epsilons: np.ndarray, result: RegimeResult, output_path: Path, title: str
) -> None:
    """Save FI degradation versus epsilon."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epsilons, result.fi_ratio_mean, color=PLOT_COLORS[0], linewidth=2)
    ax.fill_between(
        epsilons,
        result.fi_ratio_mean - result.fi_ratio_std,
        result.fi_ratio_mean + result.fi_ratio_std,
        color=PLOT_COLORS[1],
        alpha=0.2,
    )
    ax.set_xlabel("Mismatch strength epsilon")
    ax.set_ylabel("FI_imperfect / FI_ideal")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_crb_curves(
    defect_distances: np.ndarray,
    epsilons: np.ndarray,
    result: RegimeResult,
    output_path: Path,
    title: str,
) -> None:
    """Save CRB versus defect distance for selected mismatch strengths."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for color, epsilon in zip(PLOT_COLORS, PLOT_EPSILONS, strict=False):
        epsilon_index = int(np.where(np.isclose(epsilons, epsilon))[0][0])
        ax.plot(
            defect_distances,
            result.crb_mean[epsilon_index],
            label=f"epsilon={epsilon:.2f}",
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            defect_distances,
            result.crb_mean[epsilon_index] - result.crb_std[epsilon_index],
            result.crb_mean[epsilon_index] + result.crb_std[epsilon_index],
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("Defect distance delta z")
    ax.set_ylabel("CRB")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_mode_redistribution(
    measured_basis: list[dict[str, np.ndarray]], output_path: Path
) -> None:
    """Save a representative mode-redistribution plot in the unitary regime."""
    coordinates = symmetric_three_source_coordinates(
        x_separation=X_SEPARATION,
        z_displacement=REPRESENTATIVE_Z,
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for color, epsilon in zip(PLOT_COLORS, PLOT_EPSILONS, strict=False):
        rng = np.random.default_rng(seed=77 + int(100 * epsilon))
        mixing = random_orthogonal_matrix(len(measured_basis), epsilon, rng)
        probability, _, _ = imperfect_probability_and_fi(
            coordinates=coordinates,
            measurement_basis=measured_basis,
            mode_mixing_matrix=mixing,
            include_loss_channel=False,
        )
        ax.plot(
            np.arange(probability.shape[1]),
            probability[0],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"epsilon={epsilon:.2f}",
        )

    ax.set_xlabel("Mode index")
    ax.set_ylabel("Probability")
    ax.set_title("Mode power redistribution under unitary mixing")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_regime_arrays(
    output_path: Path,
    epsilons: np.ndarray,
    defect_distances: np.ndarray,
    result: RegimeResult,
) -> None:
    """Persist arrays for later inspection."""
    np.savez(
        output_path,
        epsilons=epsilons,
        defect_distances=defect_distances,
        fi_ratio_mean=result.fi_ratio_mean,
        fi_ratio_std=result.fi_ratio_std,
        crb_mean=result.crb_mean,
        crb_std=result.crb_std,
    )


def main() -> None:
    """Run the imperfection study and save arrays and plots."""
    measured_basis = hermite_gaussian_basis_x_only.measurement_basis(MEASURED_MODES)

    unitary_result = aggregate_regime(
        epsilons=EPSILONS,
        defect_distances=DEFECT_DISTANCES,
        n_realizations=N_REALIZATIONS,
        measured_basis=measured_basis,
        mixing_factory=lambda epsilon, rng: random_orthogonal_matrix(
            len(measured_basis), epsilon, rng
        ),
        lossy=False,
    )
    lossy_result = aggregate_regime(
        epsilons=EPSILONS,
        defect_distances=DEFECT_DISTANCES,
        n_realizations=N_REALIZATIONS,
        measured_basis=measured_basis,
        mixing_factory=lambda epsilon, rng: lossy_truncation_matrix(
            len(measured_basis), PADDED_MODES, epsilon, rng
        ),
        lossy=True,
    )

    save_regime_arrays(
        OUTPUT_DIRECTORY / "unitary_results.npz",
        epsilons=EPSILONS,
        defect_distances=DEFECT_DISTANCES,
        result=unitary_result,
    )
    save_regime_arrays(
        OUTPUT_DIRECTORY / "lossy_results.npz",
        epsilons=EPSILONS,
        defect_distances=DEFECT_DISTANCES,
        result=lossy_result,
    )

    plot_fi_ratio(
        epsilons=EPSILONS,
        result=unitary_result,
        output_path=OUTPUT_DIRECTORY / "unitary_fi_degradation.pdf",
        title="FI degradation under unitary mode mixing",
    )
    plot_crb_curves(
        defect_distances=DEFECT_DISTANCES,
        epsilons=EPSILONS,
        result=unitary_result,
        output_path=OUTPUT_DIRECTORY / "unitary_crb_vs_defect_distance.pdf",
        title="CRB versus defect distance under unitary mode mixing",
    )
    plot_fi_ratio(
        epsilons=EPSILONS,
        result=lossy_result,
        output_path=OUTPUT_DIRECTORY / "lossy_fi_degradation.pdf",
        title="FI degradation under truncation loss",
    )
    plot_crb_curves(
        defect_distances=DEFECT_DISTANCES,
        epsilons=EPSILONS,
        result=lossy_result,
        output_path=OUTPUT_DIRECTORY / "lossy_crb_vs_defect_distance.pdf",
        title="CRB versus defect distance with truncation loss",
    )
    plot_mode_redistribution(
        measured_basis=measured_basis,
        output_path=OUTPUT_DIRECTORY / "unitary_mode_power_redistribution.pdf",
    )


if __name__ == "__main__":
    main()
