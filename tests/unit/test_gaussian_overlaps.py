"""Testing closed-form expressions for Gaussian overlaps.

We can calculate closed-form expressions for Gaussian integrals and their derivatives
with arbitrary 3D displacement. This is beneficial since it speeds up the computation
(no integration necessary) and has higher accuracy
(analytical vs numerical calculations).

Here we make sure the closed-form expression are correct by comparing the results
to values calculated in a separate mathematica notebook.
"""

from rayfish import gaussian_overlaps

TESTSTING_CONSTANTS = {
    "s1": 1.3,
    "s2": 0.7,
    "t1": 0.93,
    "t2": 1.17,
    "p1": 0.23,
    "p2": 0.0654,
    "k": 2,
    "zR": 1,
}


def test_delta_diagonal() -> None:
    """Testing Gaussian Beam is normalized to 1."""
    expected = 1
    actual = gaussian_overlaps.delta(
        s1=TESTSTING_CONSTANTS["s1"],
        s2=TESTSTING_CONSTANTS["s2"],
        t1=TESTSTING_CONSTANTS["t1"],
        t2=TESTSTING_CONSTANTS["t2"],
        p1=TESTSTING_CONSTANTS["p1"],
        p2=TESTSTING_CONSTANTS["p2"],
        k=TESTSTING_CONSTANTS["k"],
        zr=TESTSTING_CONSTANTS["zr"],
    )

    print(actual)
    assert expected == actual


if __name__ == "__main__":
    test_delta_diagonal()
