"""Validation of ``Interpolator1DPCP`` (piecewise-constant, left-continuous, flat extrapolation).

Run from the repository root with either

    python -m pytest tests -q          # if pytest is installed
    python tests/test_interpolator_pcp.py

The analytic gradients are proven against a bump-and-reval (B&R) reference,
which is the validation the assignment asks for.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fixedincomelib import (  # noqa: E402  (import after sys.path setup)
    qfCreate1DInterpolator,
    qfInterpolate1D,
    qfInterpolate1DGrad,
    qfInterpolate1DIntegral,
    qfInterpolate1DIntegralGrad,
)

INTERP = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
EXTRAP = 'FLAT'
AXIS = [1.0, 3.0, 5.0, 7.0]
VALUES = [3.0, 4.0, 5.0, 6.0]
BUMP = 1e-4
# A forward difference of a function that is exactly affine in the ordinates
# is exact up to floating-point round-off of (f(y + h) - f(y)) / h.
BR_TOL = 1e-8


def make_interpolator(axis=AXIS, values=VALUES):
    return qfCreate1DInterpolator(list(axis), list(values), INTERP, EXTRAP)


def bump_and_reval(evaluate, axis, values, bump=BUMP):
    """Generic one-sided B&R: d evaluate / d values_i for every knot i.

    ``evaluate`` maps an interpolator to a scalar. The ordinates are copied so
    the caller's list is never mutated.
    """
    base = evaluate(make_interpolator(axis, values))
    grad = []
    for i in range(len(values)):
        bumped = [v + bump if j == i else v for j, v in enumerate(values)]
        grad.append((evaluate(make_interpolator(axis, bumped)) - base) / bump)
    return np.array(grad)


# --------------------------------------------------------------------------- #
# interpolation
# --------------------------------------------------------------------------- #
def test_interpolation_matches_docstring_convention():
    interp = make_interpolator()
    expected = {0.5: 3.0, 1.0: 3.0, 1.5: 4.0, 3.0: 4.0, 5.5: 6.0, 6.5: 6.0, 8.0: 6.0}
    for x, fx in expected.items():
        assert qfInterpolate1D(x, interp) == fx, x


def test_left_continuity_at_every_knot():
    interp = make_interpolator()
    eps = 1e-12
    for knot, value in zip(AXIS, VALUES):
        assert qfInterpolate1D(knot, interp) == value
        assert qfInterpolate1D(knot - eps, interp) == value          # limit from the left
    # ... and jumps immediately to the right of every interior knot
    for knot, next_value in zip(AXIS[:-1], VALUES[1:]):
        assert qfInterpolate1D(knot + eps, interp) == next_value


def test_flat_extrapolation_far_out():
    interp = make_interpolator()
    assert qfInterpolate1D(-1e9, interp) == VALUES[0]
    assert qfInterpolate1D(1e9, interp) == VALUES[-1]


def test_interpolation_returns_python_float_for_integer_inputs():
    interp = qfCreate1DInterpolator([1, 3, 5, 7], [3, 4, 5, 6], INTERP, EXTRAP)
    value = qfInterpolate1D(1.5, interp)
    assert isinstance(value, float) and value == 4.0


def test_single_knot_is_a_constant_function():
    interp = make_interpolator([2.0], [7.0])
    for x in (-10.0, 2.0, 10.0):
        assert qfInterpolate1D(x, interp) == 7.0
    assert qfInterpolate1DIntegral(0.0, 3.0, interp) == 21.0
    np.testing.assert_allclose(qfInterpolate1DGrad(5.0, interp), [1.0])
    np.testing.assert_allclose(qfInterpolate1DIntegralGrad(0.0, 3.0, interp), [3.0])


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #
INTEGRATION_CASES = [
    ((0.5, 0.9), 1.2),     # both inside the left wing
    ((0.5, 1.2), 2.3),     # left wing into the first bucket
    ((0.5, 3.2), 10.5),    # left wing across into the middle
    ((1.5, 5.2), 17.2),    # entirely inside the node range
    ((3.5, 7.2), 20.7),    # middle bucket out into the right wing
    ((6.0, 7.2), 7.2),     # last bucket into the right wing
    ((8.0, 10.0), 12.0),   # both inside the right wing
    ((0.1, 10.0), 50.7),   # spanning everything
    ((1.0, 3.0), 8.0),     # exactly one bucket, endpoints on knots
    ((3.0, 3.0), 0.0),     # degenerate range
]


def test_integration_matches_hand_computed_values():
    interp = make_interpolator()
    for (lo, hi), expected in INTEGRATION_CASES:
        assert abs(qfInterpolate1DIntegral(lo, hi, interp) - expected) < 1e-12, (lo, hi)


def test_integration_is_antisymmetric():
    interp = make_interpolator()
    for (lo, hi), _ in INTEGRATION_CASES:
        forward = qfInterpolate1DIntegral(lo, hi, interp)
        backward = qfInterpolate1DIntegral(hi, lo, interp)
        assert abs(forward + backward) < 1e-12, (lo, hi)


def test_integration_is_additive_over_adjacent_ranges():
    interp = make_interpolator()
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b, c = np.sort(rng.uniform(-2.0, 10.0, size=3))
        whole = qfInterpolate1DIntegral(a, c, interp)
        split = qfInterpolate1DIntegral(a, b, interp) + qfInterpolate1DIntegral(b, c, interp)
        assert abs(whole - split) < 1e-10


def test_integration_agrees_with_brute_force_riemann_sum():
    interp = make_interpolator()
    lo, hi, n_cells = -1.0, 9.0, 2_000
    width = (hi - lo) / n_cells
    midpoints = lo + (np.arange(n_cells) + 0.5) * width
    riemann = sum(qfInterpolate1D(x, interp) for x in midpoints) * width
    # midpoint rule: the error is O(jump size * cell width) at each of the knots
    assert abs(qfInterpolate1DIntegral(lo, hi, interp) - riemann) < 0.05


def test_duplicate_knots_give_zero_width_bucket():
    interp = make_interpolator([1.0, 3.0, 3.0, 7.0], [3.0, 4.0, 5.0, 6.0])
    assert qfInterpolate1D(3.0, interp) == 4.0          # first knot with x_i >= 3
    assert qfInterpolate1D(3.0 + 1e-12, interp) == 6.0  # the zero-width bucket is never hit
    np.testing.assert_allclose(qfInterpolate1DIntegralGrad(0.0, 8.0, interp), [1.0, 2.0, 0.0, 5.0])


# --------------------------------------------------------------------------- #
# gradients: analytic vs bump-and-reval
# --------------------------------------------------------------------------- #
def test_gradient_wrt_ordinate_is_one_hot():
    interp = make_interpolator()
    expected_index = {0.5: 0, 1.0: 0, 1.5: 1, 3.0: 1, 5.5: 3, 6.5: 3, 8.0: 3}
    for x, k in expected_index.items():
        grad = qfInterpolate1DGrad(x, interp)
        assert grad.shape == (len(AXIS),)
        np.testing.assert_array_equal(grad, np.eye(len(AXIS))[k])


def test_gradient_wrt_ordinate_matches_bump_and_reval():
    interp = make_interpolator()
    for x in (0.5, 1.0, 1.5, 3.0, 5.0, 5.5, 6.5, 7.0, 8.0):
        analytic = qfInterpolate1DGrad(x, interp)
        reference = bump_and_reval(lambda it, x=x: qfInterpolate1D(x, it), AXIS, VALUES)
        assert np.max(np.abs(analytic - reference)) < BR_TOL, x


def test_integral_gradient_is_overlap_length_per_bucket():
    interp = make_interpolator()
    np.testing.assert_allclose(qfInterpolate1DIntegralGrad(0.5, 3.2, interp), [0.5, 2.0, 0.2, 0.0])
    np.testing.assert_allclose(qfInterpolate1DIntegralGrad(3.5, 7.2, interp), [0.0, 0.0, 1.5, 2.2])
    np.testing.assert_allclose(qfInterpolate1DIntegralGrad(8.0, 10.0, interp), [0.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(qfInterpolate1DIntegralGrad(3.2, 0.5, interp),
                               [-0.5, -2.0, -0.2, 0.0])


def test_integral_gradient_matches_bump_and_reval():
    interp = make_interpolator()
    for (lo, hi), _ in INTEGRATION_CASES:
        analytic = qfInterpolate1DIntegralGrad(lo, hi, interp)
        reference = bump_and_reval(lambda it, lo=lo, hi=hi: qfInterpolate1DIntegral(lo, hi, it),
                                   AXIS, VALUES)
        assert np.max(np.abs(analytic - reference)) < BR_TOL, (lo, hi)


def test_integral_gradient_matches_bump_and_reval_on_random_knots():
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(1, 8))
        axis = list(np.sort(rng.uniform(-5.0, 5.0, size=n)))
        values = list(rng.normal(size=n))
        lo, hi = rng.uniform(-7.0, 7.0, size=2)
        interp = make_interpolator(axis, values)
        analytic = qfInterpolate1DIntegralGrad(lo, hi, interp)
        reference = bump_and_reval(lambda it, lo=lo, hi=hi: qfInterpolate1DIntegral(lo, hi, it),
                                   axis, values)
        assert np.max(np.abs(analytic - reference)) < 1e-6, (axis, lo, hi)
        # gradient recovers the integral by linearity: I = grad . y
        assert abs(np.dot(analytic, values) - qfInterpolate1DIntegral(lo, hi, interp)) < 1e-10


# --------------------------------------------------------------------------- #
# input validation and immutability
# --------------------------------------------------------------------------- #
def test_inputs_are_copied_not_aliased():
    axis, values = [1.0, 3.0], [3.0, 4.0]
    interp = make_interpolator(axis, values)
    values[0] = 100.0
    assert qfInterpolate1D(0.0, interp) == 3.0


def test_rejects_bad_scalar_inputs():
    interp = make_interpolator()
    for bad in (float('nan'), float('inf')):
        for call in (lambda bad=bad: qfInterpolate1D(bad, interp),
                     lambda bad=bad: qfInterpolate1DIntegral(0.0, bad, interp),
                     lambda bad=bad: qfInterpolate1DGrad(bad, interp)):
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {bad}")
    for not_a_number in ("abc", "3", True, None):
        try:
            qfInterpolate1D(not_a_number, interp)
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for {not_a_number!r}")


def test_direct_construction_rejects_bad_knot_arrays():
    from fixedincomelib.utilities.numerics import ExtrapMethod, Interpolator1DPCP
    bad_inputs = (
        (np.array([]), np.array([])),                      # no knots
        (np.array([1.0, np.nan]), np.array([1.0, 2.0])),   # non-finite knot
        (np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])),  # not one-dimensional
        (np.array([1.0, 2.0]), np.array([1.0])),           # length mismatch
        (np.array([2.0, 1.0]), np.array([1.0, 2.0])),      # decreasing knots
    )
    for axis, values in bad_inputs:
        try:
            Interpolator1DPCP(axis, values, ExtrapMethod.FLAT)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {axis!r}, {values!r}")


def test_zero_dimensional_numpy_arrays_behave_like_scalars():
    interp = make_interpolator()
    assert qfInterpolate1D(np.array(1.5), interp) == 4.0
    assert qfInterpolate1DIntegral(np.array(0.5), np.array(3.2), interp) == 10.5


def test_factory_rejects_decreasing_axis_and_length_mismatch():
    for axis, values in (([3.0, 1.0], [1.0, 2.0]), ([1.0, 2.0], [1.0])):
        try:
            qfCreate1DInterpolator(axis, values, INTERP, EXTRAP)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("expected the factory to reject the input")


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
