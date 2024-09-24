"""Replicating the results for estimating distance between 2 point sources.

This module creates 2 point sources positioned along the x-axis. We will replicate the
known results:
[1] M. Tsang, R. Nair, and X.-M. Lu, “Quantum theory of superresolution for two
incoherent optical point sources,” Phys. Rev. X 6, 031033 (2016).

It also serves as a template for now. Follow the same steps wand define the problem
how you need to get out parameters of interest to you.
"""

# Public API
__all__ = ("number_of_point_sources",)


def number_of_point_sources() -> int:
    """Define number of point sources.

    While in general this will be a user input at runtime. We hard-code it here since
    it is meant to static.
    """
    return 2
