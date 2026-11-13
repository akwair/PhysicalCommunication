"""Physical/statistical checks for external EPA/EVA profile validation."""
import importlib.util

import numpy as np
from numpy.testing import assert_allclose


def module():
    assert importlib.util.find_spec('standard_channel_study') is not None, 'standard-channel implementation missing'
    import standard_channel_study
    return standard_channel_study


def test_standard_profile_units_power_and_delay_spread():
    study = module()
    profiles = study.standard_profiles()
    for name, count, last, rms in [('EPA', 7, 410, 45), ('EVA', 9, 2510, 357)]:
        delays, powers = profiles[name]
        assert len(delays) == count
        assert_allclose(delays[-1] * study.SAMPLE_NS, last)
        assert_allclose(powers.sum(), 1, atol=1e-15)
        ns = delays * study.SAMPLE_NS
        calculated = np.sqrt(np.sum(powers * (ns - np.sum(powers * ns)) ** 2))
        assert abs(calculated/rms - 1) < .05
        assert delays.max() <= 12
        assert_allclose(delays*25, np.rint(delays*25), atol=1e-12)


def test_standard_profiles_finite_burst_physics():
    audit = module().waveform_audit()
    assert set(audit) == {'EPA', 'EVA'}
    for values in audit.values():
        assert values['max_cp_signal_difference'] < 2e-12
        assert values['max_frequency_response_error'] < 2e-12


def test_generated_noise_means_match_independent_theory():
    study = module()
    from cp_confirmation import generate_cp
    from stage3_study import projected_statistics
    from waveform_model import projection_noise_bias
    from pilot_ofdm import PILOTS
    rng = np.random.default_rng(17777)
    variance = .1
    for delays, powers in study.standard_profiles().values():
        _, yp, _, _, cp = generate_cp(rng, 4000, delays, powers, variance)
        residual = projected_statistics(yp)[1].mean(axis=1)
        expected = variance + projection_noise_bias(delays, powers, PILOTS, 12)
        assert abs(residual.mean() - expected) < 5*residual.std(ddof=1)/np.sqrt(len(cp))
        assert abs(cp.mean() - variance) < 5*cp.std(ddof=1)/np.sqrt(len(cp))


def test_paired_summary_gain_sign_and_ber_units():
    from analyze_standard_channels import paired_interval
    reference = np.array([[2., 4., 6., 8.], [1., 4., 2., 8.], [4., 5., 3., 7.]])
    candidate = reference.copy()
    candidate[0] /= 2
    candidate[2] -= 1
    answer = paired_interval(candidate, reference, [2, 2], np.random.default_rng(5), replicates=200)
    assert_allclose(answer['gain_db'], 10*np.log10(2))
    assert_allclose(answer['ci95'], [10*np.log10(2)]*2)
    assert_allclose(answer['ber_reduction'], 1/96)
    assert_allclose(answer['ber_ci95'], [1/96]*2)


def test_complete_run_and_aggregation_preserve_all_cells(tmp_path):
    from types import SimpleNamespace
    import json
    from analyze_standard_channels import run as aggregate
    study = module()
    out = tmp_path/'run'
    study.run(SimpleNamespace(frames=2, seed=17881, out=out))
    aggregate([out], tmp_path/'summary')
    report = json.loads((tmp_path/'summary'/'summary.json').read_text())
    assert report['total_groups'] == 12
    assert len(report['results']) == 6*7
    assert len(report['noise']) == 6*3
    assert len(report['comparisons']) == 6*4
    assert all(row['groups'] == 2 and 0 <= row['ber'] <= 1 for row in report['results'])
    assert not (out/'checkpoint.json').exists()
