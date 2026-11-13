"""Scientific checks for stage-2 exploratory estimators."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from noise_study import (M, statistics, fourier, priors, mix_estimate,
                         integrated_mix, sure_risk, soft_threshold, diagonal_em)
from pilot_ofdm import cn, estimate, pdp, PILOTS


def test_residual_variance_and_dof():
    rng = np.random.default_rng(871)
    n, length, v = 60000, 12, .2
    h = cn(rng, (n, length))/np.sqrt(length)
    y = h @ fourier(np.arange(length))[PILOTS].T+np.sqrt(v)*cn(rng, (n, M))
    _, rss, _ = statistics(y, length)
    ratios = rss/((M-length)*v)
    assert abs(ratios.mean()-1) < .01
    assert abs(ratios.var()-1/(M-length)) < .01
    with pytest.raises(ValueError):
        statistics(y, M)


def test_diagonal_estimator_equals_full_covariance():
    rng = np.random.default_rng(872)
    y = cn(rng, (100, M))
    z, _, basis = statistics(y, 12)
    simple = mix_estimate(z, .17, pdp(2)[None, :]) @ basis.T
    np.testing.assert_allclose(simple, estimate(y, pdp(2), .17), atol=1e-12)


def test_fixed_threshold_sure_is_unbiased():
    rng = np.random.default_rng(873)
    truth = np.array([0, 1+1j, .2, 0, .5j])
    vz, threshold = .2, .4
    z = truth+np.sqrt(vz)*cn(rng, (100000, len(truth)))
    risk = sure_risk(z, vz, threshold).mean()
    actual = np.mean(np.sum(abs(soft_threshold(z, threshold)-truth)**2, axis=1))
    assert abs(risk-actual)/actual < .02


def test_fractional_residual_bias_matches_projection_formula():
    rng = np.random.default_rng(874)
    f = fourier([0, 7.35, 11.2])[PILOTS]
    power, v, n, length = np.array([.5, .3, .2]), .01, 50000, 12
    a = fourier(np.arange(length))[PILOTS]
    projection = np.eye(M)-a @ a.conj().T/M
    expected = v+np.sum(abs(projection @ f)**2*power)/(M-length)
    y = (cn(rng, (n, 3))*np.sqrt(power)) @ f.T+np.sqrt(v)*cn(rng, (n, M))
    _, rss, _ = statistics(y, length)
    assert abs((rss/(M-length)).mean()/expected-1) < .02


def test_noise_integral_against_adaptive_quadrature():
    rng = np.random.default_rng(875)
    length, dof = 15, 1
    bank = priors(length)
    z = cn(rng, (1, length))*.2
    rss = np.array([.1])
    actual = integrated_mix(z, rss, dof, bank, nodes=96)[0]
    # Independently integrate log variance on a large interval.
    def parts(t):
        v = np.exp(t)
        total = bank+v/M
        logs = -dof*t-rss[0]/v-np.sum(np.log(total)+abs(z[0])**2/total, axis=1)
        return logs, bank/total
    grid = np.linspace(-15, 8, 1000)
    scale = max(logsumexp(parts(t)[0]) for t in grid)
    denominator = quad(lambda t: np.exp(logsumexp(parts(t)[0])-scale), -25, 20, epsabs=1e-10)[0]
    reference = []
    for j in range(length):
        def integrand(t):
            logs, shrink = parts(t)
            return np.sum(np.exp(logs-scale)*shrink[:, j])
        reference.append(z[0, j]*quad(integrand, -25, 20, epsabs=1e-10)[0]/denominator)
    np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=2e-3)


def test_em_iteration_budget_is_sufficient_on_check_sample():
    rng = np.random.default_rng(876)
    z = cn(rng, (30, 12))*.2
    rss = rng.gamma(4, .1, 30)
    # Compare optimized noise stability, not implementation-mirroring iteration values.
    e200, v200 = diagonal_em(z, rss, rss/4, 200)
    e1000, v1000 = diagonal_em(z, rss, rss/4, 1000)
    assert np.mean(abs(e200-e1000)**2) < 1e-5
    assert np.max(abs(v200-v1000)/v1000) < .05


def test_cut_noise_integral_against_adaptive_quadrature():
    rng = np.random.default_rng(877)
    z = cn(rng, (1, 15))*.2
    rss, dof = .1, 1
    bank = priors(15)
    actual = integrated_mix(z, np.array([rss]), dof, bank, 128, cut=True)[0]
    reference = []
    for j in range(15):
        def integrand(t):
            v = np.exp(t)
            estimate = mix_estimate(z, v, bank)[0, j]/z[0, j]
            return estimate.real*np.exp(-dof*t-rss/v)*rss
        reference.append(z[0, j]*quad(integrand, -25, 25, epsabs=1e-10)[0])
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-3)


def test_cp_time_domain_matches_integer_delay_frequency_model():
    rng = np.random.default_rng(878)
    x = cn(rng, (64,))
    h = cn(rng, (12,))/np.sqrt(12)
    tx = np.fft.ifft(x, norm='ortho')
    cp = 16
    rx = np.convolve(np.concatenate([tx[-cp:], tx]), h)[cp:cp+64]
    np.testing.assert_allclose(np.fft.fft(rx, norm='ortho'), x*np.fft.fft(h, 64), atol=1e-12)
