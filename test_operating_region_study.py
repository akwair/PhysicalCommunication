import numpy as np
from operating_region_study import geometry, prediction, simulate
from waveform_model import projection_noise_bias


def test_matched_gamma_moments_and_null_crossover():
    for m in (16, 32):
        for length in (12, 15):
            p = prediction(m, length, 0, 20)
            assert p['bias'] == 0
            assert p['crossover_blocks'] is None
            assert np.isclose(p['normalized_mse']['16'], 1/(16*(m-length)))


def test_direct_eigenvalue_moments_and_independent_existing_bias_code():
    for m in (16, 32):
        for length in (12, 15):
            for delta in (.2, .5, .9):
                q, _, s = geometry(m, length, delta)
                p = prediction(m, length, delta, 20)
                ev = np.linalg.eigvalsh(s + .01*q)
                assert np.isclose(ev.sum()/(m-length), .01+p['bias'])
                assert np.isclose(np.sum(ev**2)/(m-length)**2, p['single_block_variance'])
                b = projection_noise_bias([0, 7+delta, 11+delta], [.5,.3,.2],
                                          np.arange(0,64,64//m), length)
                assert np.isclose(p['bias'], b)


def test_noise_simulation_recovers_matched_distribution():
    vals = simulate(16, 12, 0, 65803, groups=12000)[:, 2]
    # Independent observed-data check, 192000 pilot blocks.
    expected_var = .01**2/64
    assert abs(vals.mean()-.01) < 5*np.sqrt(expected_var/len(vals))
    assert abs(vals.var(ddof=1)/expected_var - 1) < .05


def test_nested_windows_remove_raw_signal_energy():
    for m in (16, 32):
        for delta in (.2,.5,.9):
            energy = [prediction(m,l,delta,20)['raw_leakage'] for l in range(12,16)]
            assert np.all(np.diff(energy) <= 1e-12)
