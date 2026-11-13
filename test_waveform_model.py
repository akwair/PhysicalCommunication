"""Independent finite-CP checks of the ideal signed-baseband OFDM model."""
import numpy as np
import pytest

from waveform_model import (signed_frequencies, delay_atoms, channel_response,
                            modulate, receive_ofdm, projection_noise_bias)


def complex_normal(rng, shape):
    return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))/np.sqrt(2)


def test_signed_frequency_order_including_nyquist():
    np.testing.assert_allclose(signed_frequencies(8), [0, .125, .25, .375, -.5, -.375, -.25, -.125])
    np.testing.assert_allclose(signed_frequencies(7), np.array([0, 1, 2, 3, -3, -2, -1])/7)


@pytest.mark.parametrize('n_fft', [63, 64])
def test_oversampled_modulator_has_unitary_base_rate_scaling(n_fft):
    rng = np.random.default_rng(1480)
    symbols = complex_normal(rng, (3, n_fft))
    waveform = modulate(symbols, cp_samples=0, oversample=20).reshape(3, n_fft*20)
    np.testing.assert_allclose(waveform[:, ::20], np.fft.ifft(symbols, norm='ortho'), atol=1e-12)
    np.testing.assert_allclose(np.mean(abs(waveform)**2, axis=-1), np.mean(abs(symbols)**2, axis=-1), atol=1e-12)


@pytest.mark.parametrize('cp', [11, 16])
def test_integer_delay_agrees_with_base_rate_linear_convolution(cp):
    rng = np.random.default_rng(1481)
    symbols = complex_normal(rng, (3, 64))
    delays = np.array([0, 7, 11])
    gains = complex_normal(rng, 3)
    taps = np.zeros(12, dtype=complex)
    taps[delays] = gains
    time_symbols = np.fft.ifft(symbols, norm='ortho')
    transmit = np.concatenate([time_symbols[:, -cp:], time_symbols], axis=-1).reshape(-1)
    convolution = np.convolve(transmit, taps)
    windows = np.arange(3)[:, None]*(64+cp)+cp+np.arange(64)
    reference = np.fft.fft(convolution[windows], norm='ortho')
    actual = receive_ofdm(symbols, delays, gains, cp_samples=cp, oversample=20)
    np.testing.assert_allclose(actual, reference, atol=2e-12)
    np.testing.assert_allclose(actual, symbols*channel_response(delays, gains), atol=2e-12)


@pytest.mark.parametrize('n_fft', [63, 64])
def test_fractional_paths_match_signed_frequency_model_for_independent_symbols(n_fft):
    rng = np.random.default_rng(1482)
    symbols = complex_normal(rng, (4, 3, n_fft))
    delays = np.array([0, 7.35, 11.2])
    gains = complex_normal(rng, (4, 3))*np.sqrt([.5, .3, .2])
    expected = symbols*channel_response(delays, gains, n_fft=n_fft)[:, None, :]
    actual = receive_ofdm(symbols, delays, gains, cp_samples=16, oversample=20)
    np.testing.assert_allclose(actual, expected, atol=4e-12)


def test_fractional_negative_frequency_tone_detects_unsigned_convention_error():
    symbols = np.zeros((2, 64), dtype=complex)
    symbols[:, 48] = 1  # physical baseband frequency -1/4, not +3/4.
    actual = receive_ofdm(symbols, [.5], [1], cp_samples=1, oversample=2)
    np.testing.assert_allclose(actual[:, 48], np.exp(1j*np.pi/4), atol=1e-12)
    wrong = np.exp(-2j*np.pi*(48/64)*.5)
    assert np.max(abs(actual[:, 48]-wrong)) > 1.9


def test_cp_insufficiency_creates_previous_symbol_dependence():
    rng = np.random.default_rng(1483)
    symbols = complex_normal(rng, (3, 64))
    alternate = symbols.copy()
    alternate[0] *= -1
    delays, gains = [0, 7.35, 11.2], [1, .4j, .5]
    good = receive_ofdm(symbols, delays, gains, cp_samples=16, oversample=20)
    good_alternate = receive_ofdm(alternate, delays, gains, cp_samples=16, oversample=20)
    np.testing.assert_allclose(good[1], good_alternate[1], atol=1e-12)
    bad = receive_ofdm(symbols, delays, gains, cp_samples=8, oversample=20)
    bad_alternate = receive_ofdm(alternate, delays, gains, cp_samples=8, oversample=20)
    assert np.linalg.norm(bad[1]-bad_alternate[1]) > .1
    predicted = symbols*channel_response(delays, gains)
    assert np.linalg.norm(bad[1]-predicted[1])/np.linalg.norm(predicted[1]) > .01


def test_delays_off_oversampling_grid_are_rejected_instead_of_rounded():
    with pytest.raises(ValueError, match='grid'):
        receive_ofdm(np.ones((2, 64)), [7.35], [1], oversample=8)


@pytest.mark.parametrize('length', [12, 15])
def test_comb_pilot_residual_bias_is_invariant_to_signed_convention(length):
    pilots, delays, powers = np.arange(0, 64, 4), [0, 7.35, 11.2], [.5, .3, .2]
    signed = projection_noise_bias(delays, powers, pilots, length)
    # Independent projection in the original unsigned convention.
    a = np.exp(-2j*np.pi*np.outer(pilots/64, np.arange(length)))
    b = np.exp(-2j*np.pi*np.outer(pilots/64, delays))
    q = np.eye(len(pilots))-a@np.linalg.pinv(a)
    unsigned = np.sum(abs(q@b)**2*np.array(powers))/(len(pilots)-length)
    np.testing.assert_allclose(signed, unsigned, atol=2e-14)
    assert signed > .02


def test_signed_and_unsigned_atoms_agree_for_integer_delays():
    delays = np.arange(12)
    unsigned = np.exp(-2j*np.pi*np.outer(np.arange(64)/64, delays))
    np.testing.assert_allclose(delay_atoms(delays), unsigned, atol=2e-14)


def test_projection_bias_formula_matches_signed_model_monte_carlo():
    rng = np.random.default_rng(1484)
    pilots, delays, powers = np.arange(0, 64, 4), [0, 7.35, 11.2], [.5, .3, .2]
    count, length, variance = 25000, 12, .01
    a = delay_atoms(np.arange(length))[pilots]
    gains = complex_normal(rng, (count, 3))*np.sqrt(powers)
    y = channel_response(delays, gains)[:, pilots]+np.sqrt(variance)*complex_normal(rng, (count, len(pilots)))
    residual = y-(y@a.conj()/len(pilots))@a.T
    observed = np.mean(np.sum(abs(residual)**2, axis=1)/(len(pilots)-length))
    expected = variance+projection_noise_bias(delays, powers, pilots, length)
    assert abs(observed/expected-1) < .025
