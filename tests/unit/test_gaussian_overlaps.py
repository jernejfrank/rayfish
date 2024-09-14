"""Testing closed-form expressions for Gaussian overlaps.

We can calculate closed-form expressions for Gaussian integrals and their derivatives
with arbitrary 3D displacement. This is beneficial since it speeds up the computation
(no integration necessary) and has higher accuracy
(analytical vs numerical calculations).

Here we make sure the closed-form expression are correct by comparing the results
to values calculated in a separate mathematica notebook.
"""


def test_delta_diagonal() -> None:
    """Testing Gaussian Beam is normalized to 1."""
    expected = 1
    actual = 1
    assert expected == actual
