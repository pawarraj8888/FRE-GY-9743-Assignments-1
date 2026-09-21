import copy
import numbers
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


def _as_finite_float(value, name: str) -> float:
    """Validate a real scalar input at the library boundary and return it as a float.

    Booleans and numeric strings are rejected on purpose: they are almost
    always a caller bug rather than a legitimate abscissa.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()                      # 0-d arrays behave like scalars
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value_ = float(value)
    if not np.isfinite(value_):
        raise ValueError(f"{name} must be finite, got {value_}")
    return value_


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6

    Convention
    ----------
    Knot i (0-based) carries its ordinate y_i on the bucket (x_{i-1}, x_i]:

        f(x) = y_0                 for x <= x_0            (flat left wing)
        f(x) = y_i                 for x_{i-1} < x <= x_i  (i = 1..N-1)
        f(x) = y_{N-1}             for x > x_{N-1}         (flat right wing)

    so f(x_i) = y_i = lim_{x -> x_i^-} f(x): the function is left-continuous
    at every knot and jumps immediately to the right of it.

    Because f is piecewise constant, every quantity is affine in the
    ordinates y:

        I[f; l, u] = sum_i w_i(l, u) * y_i,   w_i = |[l, u] ∩ bucket_i|

    hence  d f(x) / d y  is a one-hot vector on the active knot and
    d I / d y  is the vector of overlap lengths w. Both gradients are exact
    (no finite differences) and are what the bump-and-reval check in the
    notebook validates.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

        knots = np.asarray(self.axis1_, dtype=float)
        ordinates = np.asarray(self.values_, dtype=float)
        if knots.ndim != 1 or ordinates.ndim != 1:
            raise ValueError("axis1 and values must be one-dimensional")
        if len(knots) == 0:
            raise ValueError("Interpolator1DPCP needs at least one knot")
        if len(knots) != len(ordinates):
            raise ValueError("axis1 and values must have the same length")
        if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(ordinates)):
            raise ValueError("axis1 and values must be finite")
        if np.any(np.diff(knots) < 0):
            raise ValueError("axis1 must be non-decreasing")

        # Bucket i is (bucket_lower_[i], bucket_upper_[i]]; the two wings are
        # folded into the first and last bucket so that flat extrapolation
        # needs no special casing anywhere below.
        self.knots_ = knots
        self.ordinates_ = ordinates
        self.bucket_lower_ = np.concatenate(([-np.inf], knots[:-1]))
        self.bucket_upper_ = np.concatenate((knots[:-1], [np.inf]))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _knot_index(self, x: float) -> int:
        """Index of the knot whose ordinate applies at x.

        `searchsorted(..., side='left')` returns the first knot with x_i >= x,
        which is exactly the left-continuous bucket (x_{i-1}, x_i]. Anything
        beyond the last knot is clamped to it (flat right extrapolation);
        anything at or before the first knot already maps to index 0 (flat
        left extrapolation).
        """
        x_ = _as_finite_float(x, "x")
        index = int(np.searchsorted(self.knots_, x_, side='left'))
        return min(index, self.length_ - 1)

    def _overlap_lengths(self, start_x: float, end_x: float) -> np.ndarray:
        """w_i = length of [start_x, end_x] ∩ bucket_i, for start_x <= end_x."""
        lower = np.maximum(self.bucket_lower_, start_x)
        upper = np.minimum(self.bucket_upper_, end_x)
        return np.maximum(upper - lower, 0.0)

    # ------------------------------------------------------------------ #
    # Interpolator1D interface
    # ------------------------------------------------------------------ #
    def interpolate(self, x: float) -> float:
        return float(self.ordinates_[self._knot_index(x)])

    def integrate(self, start_x: float, end_x: float) -> float:
        # I[f; l, u] = sum_i w_i y_i  -- the integral is linear in the ordinates,
        # so it is the dot product of its own gradient with the ordinates.
        weights = self.gradient_of_integrated_value_wrt_ordinate(start_x, end_x)
        return float(np.dot(weights, self.ordinates_))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        # f(x) = y_k for the active knot k, so d f / d y = e_k.
        gradient = np.zeros(self.length_, dtype=float)
        gradient[self._knot_index(x)] = 1.0
        return gradient

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        # d I[f; l, u] / d y_i = w_i(l, u); an inverted range flips the sign,
        # matching the convention  int_l^u = -int_u^l.
        start_ = _as_finite_float(start_x, "start_x")
        end_ = _as_finite_float(end_x, "end_x")
        if start_ > end_:
            return -self._overlap_lengths(end_, start_) + 0.0   # + 0.0 turns -0. into 0.
        return self._overlap_lengths(start_, end_)


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')