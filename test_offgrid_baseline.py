"""Observable recovery and input-contract tests for the CPU sparse baseline."""

import importlib
import importlib.util

import numpy as np
import pytest


def estimate(*args, **kwargs):
    assert importlib.util.find_spec("offgrid_baseline") is not None, (
        "The off-grid baseline module has not been implemented"
    )
    return importlib.import_module("offgrid_baseline").estimate_offgrid(*args, **kwargs)


def response(frequencies, delays, gains):
    return np.exp(-2j * np.pi * np.outer(frequencies, delays)) @ gains


@pytest.mark.parametrize("delays", [[0.63, 3.45, 8.78], [0.0, 3.0, 9.0]])
def test_noiseless_three_path_recovery_without_supplied_order(delays):
    frequencies = np.fft.fftfreq(64)
    pilots = frequencies[::4]
    gains = np.array([1.0 + 0.2j, -0.45 + 0.3j, 0.28 - 0.35j])
    expected = response(frequencies, delays, gains)
    result = estimate(response(pilots, delays, gains), pilots, frequencies, 0.0)

    assert result["selected_order"] == 3
    np.testing.assert_allclose(result["response"], expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(result["delays"], delays, atol=2e-6)
    assert result["rss"] < 1e-12


def test_arbitrary_signed_frequencies_and_continuous_refinement():
    pilots = np.array([-.49, -.44, -.4, -.33, -.29, -.23, -.17, -.08,
                       .0, .04, .12, .19, .27, .32, .41, .47])
    frequencies = np.linspace(-.48, .48, 43)
    delays, gains = [1.37, 7.12], [1.0 + .1j, -.35 + .2j]
    y = response(pilots, delays, gains)
    refined = estimate(y, pilots, frequencies, 0.0)
    on_grid = estimate(y, pilots, frequencies, 0.0, refine=False)

    assert refined["selected_order"] == 2
    np.testing.assert_allclose(refined["response"], response(frequencies, delays, gains),
                               atol=2e-6, rtol=2e-6)
    assert refined["rss"] < on_grid["rss"] * 1e-5


def test_noise_controls_order_and_zero_order_is_valid():
    frequencies = np.fft.fftfreq(64)
    pilots = frequencies[::4]
    y = response(pilots, [2.25], [0.02 + .01j])
    result = estimate(y, pilots, frequencies, noise_variance=1.0)

    assert result["selected_order"] == 0
    np.testing.assert_array_equal(result["response"], np.zeros(64))
    assert result["delays"].size == 0
    assert result["gains"].size == 0
    assert result["rss"] == pytest.approx(np.vdot(y, y).real)


def test_noisy_result_has_finite_diagnostics_and_consistent_reconstruction():
    frequencies = np.fft.fftfreq(64)
    pilots = frequencies[::4]
    rng = np.random.default_rng(304)
    variance = .02
    y = response(pilots, [.63, 3.45, 8.78], [1, -.4 + .3j, .25j])
    y += np.sqrt(variance / 2) * (rng.normal(size=16) + 1j * rng.normal(size=16))
    result = estimate(y, pilots, frequencies, variance)

    assert 0 <= result["selected_order"] <= 6
    assert np.all(np.isfinite(result["response"]))
    assert np.all((result["delays"] >= 0) & (result["delays"] <= 12))
    np.testing.assert_allclose(result["response"],
                               response(frequencies, result["delays"], result["gains"]))
    residual = y - response(pilots, result["delays"], result["gains"])
    assert result["rss"] == pytest.approx(np.vdot(residual, residual).real)
    assert result["selected_score"] == min(result["score_history"])
    assert len(result["score_history"]) == len(result["rss_history"])
    assert result["nfev"] > 0


@pytest.mark.parametrize("bad_kwargs", [
    {"noise_variance": -1}, {"noise_variance": np.nan},
    {"max_delay": 0}, {"grid_step": 0}, {"max_paths": 1.5},
])
def test_invalid_configuration_is_rejected(bad_kwargs):
    kwargs = {"noise_variance": .1, **bad_kwargs}
    with pytest.raises(ValueError):
        estimate(np.ones(16), np.fft.fftfreq(16), np.fft.fftfreq(64), **kwargs)
