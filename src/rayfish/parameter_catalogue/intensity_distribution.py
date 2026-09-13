"""Parametrizations of the relative point-source intensities I_v.

These define the weights I_v entering the incoherent mixed state R = Σ_v I_v |ψ_v><ψ_v|,
normalized so that Σ_v I_v = 1. The ``equal`` parametrization recovers the uniform
I_v = 1/V; ``contrast`` tunes a single source relative to the others.
"""

from hamilton.function_modifiers import config

__all__ = (
    "point_source_intensities__equal",
    "point_source_intensities__contrast",
)


@config.when_not(intensity="contrast")
def point_source_intensities__equal(number_of_point_sources: int) -> list[float]:
    """Set intensities to equal for all sources."""
    return [1 / number_of_point_sources for _ in range(number_of_point_sources)]


@config.when(intensity="contrast")
def point_source_intensities__contrast(
    number_of_point_sources: int,
    intensity_contrast: float,
    target_point_source_idx: int,
) -> list[float]:
    """Tune one source's brightness relative to the others via a contrast ratio.

    ``intensity_contrast`` is the ratio ``r = I_target / I_other`` of the target
    source to each of the remaining (equally bright) sources: ``r = 1`` is equal
    brightness, ``r > 1`` makes the target brighter, ``r < 1`` dimmer (``r = 0``
    switches it off). Negative ratios are unphysical and floored to ``0``, so every
    intensity stays in ``[0, 1]`` and they sum to exactly ``1`` by construction.
    """
    r = max(intensity_contrast, 0.0)
    other = 1.0 / (r + number_of_point_sources - 1)
    target = r * other
    return [
        target if i == target_point_source_idx else other
        for i in range(number_of_point_sources)
    ]
