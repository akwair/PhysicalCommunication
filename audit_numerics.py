"""Numerical sensitivity audit across all stage-2 scenario types."""
import json
from pathlib import Path
import numpy as np
from noise_study import profiles, fourier, statistics, integrated_mix, priors, diagonal_em, M
from pilot_ofdm import cn, PILOTS

rng = np.random.default_rng(80261113)
rows = []
for profile, (delays, power) in profiles().items():
    for snr in [0, 10, 20]:
        variance = 10**(-snr/10)
        truth = (cn(rng, (120, len(power)))*np.sqrt(power)) @ fourier(delays).T
        y = truth[:, PILOTS]+np.sqrt(variance)*cn(rng, (120, M))
        for length in [12, 15]:
            z, rss, basis = statistics(y, length)
            bank = priors(length)
            for cut in [False, True]:
                e128 = integrated_mix(z, rss, M-length, bank, 128, cut) @ basis.T
                e256 = integrated_mix(z, rss, M-length, bank, 256, cut) @ basis.T
                difference_db = 10*np.log10(np.sum(abs(e128-truth)**2)/np.sum(abs(e256-truth)**2))
                assert abs(difference_db) < .005, (profile, snr, length, cut, difference_db)
                rows.append(dict(profile=profile, snr=snr, length=length, cut=cut,
                                 quadrature_nmse_change_db=float(difference_db)))
            e200, _ = diagonal_em(z, rss, rss/(M-length), 200)
            e1000, _ = diagonal_em(z, rss, rss/(M-length), 1000)
            delta_db = 10*np.log10(np.sum(abs(e200 @ basis.T-truth)**2)/np.sum(abs(e1000 @ basis.T-truth)**2))
            assert abs(delta_db) < .03, (profile, snr, length, delta_db)
            rows[-1]['em_nmse_change_db'] = float(delta_db)
Path('results/numerical_audit.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print('24 scenarios checked; quadrature <0.005 dB and EM iterations <0.03 dB effect.')
print('Maximum quadrature change:', max(abs(r['quadrature_nmse_change_db']) for r in rows))
print('Maximum EM change:', max(abs(r.get('em_nmse_change_db', 0)) for r in rows))
