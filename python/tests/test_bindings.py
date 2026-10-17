"""Tests of the Python bindings (the `pyo3` feature), run against the built wheel.

Expected values come from closed forms (an exact line, a mean, Q16.16 scaling),
not from calling the library, except for the batch functions, which must agree
with the scalar functions they wrap.
"""

import numpy as np
import pytest

import alice_edge

Q = 1 << 16
ID = alice_edge.semantics_id()


def ramp(slope, intercept, n):
    # y = slope * x + intercept at x = 0..n-1, the abscissa fit_linear assumes
    return np.array([slope * x + intercept for x in range(n)], dtype=np.int32)


def test_constants():
    assert alice_edge.Q16_SHIFT == 16
    assert alice_edge.Q16_ONE == Q


def test_fit_linear_exact_line():
    # least squares on an exact line returns that line
    assert alice_edge.fit_linear(ramp(100, 100, 5)) == (100 * Q, 100 * Q)
    assert alice_edge.fit_linear(ramp(-7, 250, 9)) == (-7 * Q, 250 * Q)


def test_fit_constant_is_the_mean():
    assert alice_edge.fit_constant(np.array([10, 20, 30], dtype=np.int32)) == 20 * Q


def test_evaluate_linear():
    # slope and intercept in Q16.16, x a plain integer
    assert alice_edge.evaluate_linear(100 * Q, 100 * Q, 4, ID) == 500 * Q


def test_compute_error_of_an_exact_fit_is_zero():
    data = ramp(3, -40, 8)
    slope, intercept = alice_edge.fit_linear(data)
    assert alice_edge.compute_error(data, slope, intercept, ID) == 0


def test_should_use_linear():
    assert alice_edge.should_use_linear(ramp(5, 0, 10)) is True
    # a constant signal: the constant model already has zero error
    assert alice_edge.should_use_linear(np.full(10, 42, dtype=np.int32)) is False
    # fewer than three samples never selects the linear model
    assert alice_edge.should_use_linear(ramp(5, 0, 2)) is False


def test_conversions():
    assert alice_edge.int_to_q16(3) == 3 * Q
    assert alice_edge.q16_to_int(3 * Q) == 3
    assert alice_edge.q16_to_f32(Q + Q // 2) == 1.5
    out = alice_edge.q16_to_f32_batch(np.array([0, Q, -2 * Q, Q // 4], dtype=np.int32))
    assert out.dtype == np.float32
    assert out.tolist() == [0.0, 1.0, -2.0, 0.25]


def test_fit_linear_batch_matches_the_scalar_fit():
    lines = [(100, 100), (-7, 250), (0, 9), (12, -3)]
    streams = np.stack([ramp(s, b, 6) for s, b in lines])
    out = alice_edge.fit_linear_batch(streams)
    assert out.shape == (len(lines), 2)
    assert [tuple(row) for row in out.tolist()] == [(s * Q, b * Q) for s, b in lines]
    for row, stream in zip(out.tolist(), streams):
        assert tuple(row) == alice_edge.fit_linear(stream)


def test_evaluate_linear_batch_matches_the_scalar_evaluation():
    coeffs = np.array([[100 * Q, 100 * Q], [-7 * Q, 250 * Q]], dtype=np.int32)
    xs = np.array([0, 1, 4, 10], dtype=np.int32)
    out = alice_edge.evaluate_linear_batch(coeffs, xs, ID)
    assert out.shape == (2, 4)
    for i, (slope, intercept) in enumerate(coeffs.tolist()):
        assert out[i].tolist() == [alice_edge.evaluate_linear(slope, intercept, int(x), ID) for x in xs]


@pytest.mark.parametrize("width", [1, 3])
def test_evaluate_linear_batch_refuses_a_coefficient_row_that_is_not_two_wide(width):
    # a row is [slope, intercept]; one column used to index past the end (a Rust panic,
    # raised as PanicException, which `except Exception` does not catch) and three
    # columns used to be read misaligned without an error
    coeffs = np.ones((2, width), dtype=np.int32)
    with pytest.raises(ValueError, match="shape"):
        alice_edge.evaluate_linear_batch(coeffs, np.array([1, 2], dtype=np.int32), ID)


def test_coefficients_with_another_identifier_are_refused_naming_both():
    foreign = bytes([ID[0] ^ 1]) + bytes(ID[1:])
    with pytest.raises(ValueError) as e:
        alice_edge.evaluate_linear(Q, Q, 1, foreign)
    assert bytes(foreign).hex() in str(e.value) and bytes(ID).hex() in str(e.value)
    with pytest.raises(ValueError):
        alice_edge.compute_error(np.array([1, 2, 3], dtype=np.int32), Q, 0, foreign)
    with pytest.raises(ValueError):
        alice_edge.evaluate_linear_batch(np.array([[Q, 0]], dtype=np.int32), np.array([1], dtype=np.int32), foreign)
