"""Concrete parameter-estimation problems.

Each submodule defines a specific point-source geometry and the coordinate Jacobian B
that re-parametrizes the raw source coordinates onto the parameters of interest, applied
to the (Q)FIM as H̃ = B H B^T. Jacobians are defined directly per problem; import the
submodule you need, e.g.::

    from rayfish.parameter_catalogue import distance_2_sources_x
"""
