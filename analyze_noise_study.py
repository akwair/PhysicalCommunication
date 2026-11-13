"""Pool independent exploratory runs and produce paired frame bootstrap intervals."""
import json
from pathlib import Path
import numpy as np


def main():
    paths = [Path(f'results/noise_rep{i}') for i in [1, 2, 3]]
    documents = [json.loads((p/'results.json').read_text()) for p in paths]
    assert len({d['source_sha256'] for d in documents}) == 1
    assert len({d['seed'] for d in documents}) == 3
    arrays = [np.load(p/'frame_metrics.npz') for p in paths]
    pooled, comparisons = [], []
    rng = np.random.default_rng(90261113)
    lookup = {}
    for row in documents[0]['results']:
        key = f"{row['profile']}_{row['snr_db']}_{row['length']}_{row['method']}"
        data = np.concatenate([a[key] for a in arrays], axis=1)
        lookup[key] = data
        pooled.append(dict(profile=row['profile'], snr_db=row['snr_db'], length=row['length'],
                           method=row['method'], frames=data.shape[1],
                           nmse_db=float(10*np.log10(data[0].sum()/data[1].sum())),
                           ber=float(data[2].sum()/(96*data.shape[1]))))
    pairs = [('mixture_integrated', 'mixture_plugin'), ('mixture_cut', 'mixture_plugin'),
             ('mixture_cut', 'mixture_integrated'), ('diagonal_eb_plugin', 'mixture_plugin')]
    # All eight profile/support cells at 10 dB are reported, including failures.
    for profile in ['exp2', 'exp8', 'sparse_integer', 'fractional_stress']:
        for length in [12, 15]:
            for candidate, reference in pairs:
                prefix = f'{profile}_10_{length}_'
                ca, re = lookup[prefix+candidate], lookup[prefix+reference]
                samples, ber_samples = [], []
                for _ in range(20):
                    indices = rng.integers(ca.shape[1], size=(30, ca.shape[1]))
                    samples.extend(10*np.log10(re[0, indices].sum(axis=1)/ca[0, indices].sum(axis=1)))
                    ber_samples.extend((re[2, indices]-ca[2, indices]).mean(axis=1)/96)
                comparisons.append(dict(profile=profile, snr_db=10, length=length,
                                        candidate=candidate, reference=reference,
                                        gain_db=float(10*np.log10(re[0].sum()/ca[0].sum())),
                                        gain_db_ci95=np.quantile(samples, [.025, .975]).tolist(),
                                        ber_reduction=float((re[2]-ca[2]).mean()/96),
                                        ber_reduction_ci95=np.quantile(ber_samples, [.025, .975]).tolist()))
    noise = []
    for row in documents[0]['noise']:
        key = f"{row['profile']}_{row['snr_db']}_{row['length']}_noise"
        values = np.concatenate([a[key] for a in arrays])
        true_variance = 10**(-row['snr_db']/10)
        noise.append(dict(profile=row['profile'], snr_db=row['snr_db'], length=row['length'],
                          mean_ratio=float(values.mean()/true_variance),
                          relative_std=float(values.std()/true_variance)))
    out = Path('results/noise_summary')
    out.mkdir(exist_ok=True)
    summary = dict(status='exploratory_intervals_no_multiple_comparison_correction',
                   seeds=[d['seed'] for d in documents], bootstrap_replicates=600,
                   results=pooled, paired_comparisons=comparisons, noise=noise)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['Stage 2 pooled results: 3 independent seeds, 15000 frames per scenario.',
             'NMSE_dB: lower is better. Gain_dB: positive favors candidate.',
             'Exploratory paired frame bootstrap; no multiplicity correction.', '',
             '10 dB NMSE table:', 'profile | L | plug-in | full integration | cut | diagonal EB | joint EM']
    methods = ['mixture_plugin', 'mixture_integrated', 'mixture_cut', 'diagonal_eb_plugin', 'diagonal_em_joint']
    for profile in ['exp2', 'exp8', 'sparse_integer', 'fractional_stress']:
        for length in [12, 15]:
            values = [next(r['nmse_db'] for r in pooled if r['profile']==profile and r['snr_db']==10
                           and r['length']==length and r['method']==m) for m in methods]
            lines.append(f'{profile} | {length} | '+' | '.join(f'{v:.3f}' for v in values))
    lines.append('\nPaired comparisons at 10 dB:')
    for c in comparisons:
        lines.append(f"{c['profile']} L={c['length']} {c['candidate']} vs {c['reference']}: "
                     f"gain={c['gain_db']:.3f} dB, CI95={c['gain_db_ci95']}")
    (out/'summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:13]))


if __name__ == '__main__':
    main()
