"""Independently regenerate observations and verify saved EPA/EVA metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from cp_confirmation import generate_cp
from standard_channel_study import standard_profiles, SNR_DB
from stage3_study import projected_statistics, metrics, M


def run():
    evidence, total_flagged = {}, 0
    for rep in (1, 2, 3):
        path = Path(f'results/standard_channel{rep}')
        doc = json.loads((path/'results.json').read_text())
        assert doc['complete'] and doc['groups_per_cell'] == 300
        for name, expected in doc['source_hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
        rng = np.random.default_rng(doc['seed'])
        with np.load(path/'frame_metrics.npz') as data, np.load(path/'capped_inputs.npz') as inputs, \
             np.load(f'results/standard_channel_budget/frame_metrics{rep}.npz') as budget:
            for profile, (delays, powers) in standard_profiles().items():
                for snr in SNR_DB:
                    prefix = f'{profile}_{snr}_'
                    truth, yp, bits, yd, cp = generate_cp(rng, 300, delays, powers, 10**(-snr/10))
                    z, residual, basis = projected_statistics(yp)
                    sources = dict(residual16=residual.mean(axis=1), cp16=cp,
                                   known_noise=np.full(300, 10**(-snr/10)))
                    assert_allclose(data[prefix+'delay_ls'], metrics(z[:, 0] @ basis.T, truth, bits, yd), atol=1e-14, rtol=1e-12)
                    for source, noise in sources.items():
                        assert_array_equal(noise, data[prefix+source+'_noise'])
                        estimate = (z[:, 0]*np.maximum(0, 1-noise[:, None]/M/
                                    np.maximum(abs(z[:, 0])**2, 1e-300))) @ basis.T
                        assert_allclose(data[prefix+'eb_'+source], metrics(estimate, truth, bits, yd), atol=1e-14, rtol=1e-12)
                        calls = next(c['calls'] for c in doc['diagnostics'] if
                            (c['profile'], c['snr_db'], c['source']) == (profile, snr, source))
                        for index, call in enumerate(calls):
                            if all(call['refinement_success']):
                                assert_array_equal(data[prefix+'refined_'+source][:, index],
                                                   budget[prefix+'refined_'+source][:, index])
                            else:
                                total_flagged += 1
                                key = prefix+str(index)
                                for suffix, values in [('_pilot', yp[index, 0]), ('_truth', truth[index]),
                                                       ('_bits', bits[index]), ('_received', yd[index]),
                                                       ('_'+source+'_noise', noise[index])]:
                                    assert_array_equal(inputs[key+suffix], values)
            assert all(np.all(np.isfinite(data[k])) for k in data.files)
            assert all(np.all(np.isfinite(budget[k])) for k in budget.files)
        for name in ('results.json', 'frame_metrics.npz', 'capped_inputs.npz'):
            file = path/name
            evidence[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    audit = json.loads(Path('results/standard_channel_budget/summary.json').read_text())
    assert total_flagged == audit['replayed_calls']
    summary = json.loads(Path('results/standard_channel_summary/summary.json').read_text())
    data = [np.load(f'results/standard_channel{i}/frame_metrics.npz') for i in (1, 2, 3)]
    for row in summary['results']:
        key = f"{row['profile']}_{row['snr_db']}_{row['method']}"
        a = np.concatenate([part[key] for part in data], axis=1)
        assert_allclose(row['nmse_db'], 10*np.log10(a[0].sum()/a[1].sum()), atol=1e-13)
        assert_allclose(row['ber'], a[2].sum()/(96*a.shape[1]), atol=1e-15)
    zscores = [abs(r['mean_ratio']-r['expected_ratio'])/r['se_mean_ratio']
               for r in summary['noise'] if r['se_mean_ratio'] > 0]
    report = dict(status='passed', groups=5400, regenerated_noise_arrays_exact=True,
                  regenerated_delay_ls_and_eb_metrics_match=True,
                  saved_flagged_inputs_match_regenerated_observations=True,
                  budget_unflagged_metrics_unchanged=True, replayed_calls=total_flagged,
                  aggregate_metrics_match_per_group_arrays=True,
                  maximum_noise_mean_error_in_standard_errors=max(zscores), evidence_sha256=evidence)
    Path('results/standard_channel_summary/verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'evidence_sha256'}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
