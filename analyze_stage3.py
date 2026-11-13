"""Paired exploratory analysis of stage-3 calibration and off-grid controls."""
import json
from pathlib import Path
import numpy as np


def summarize(prefix, output, pairs):
    paths = [Path(f'results/{prefix}{i}') for i in [1, 2, 3]]
    docs = [json.loads((p/'results.json').read_text()) for p in paths]
    assert len({d['seed'] for d in docs}) == 3
    assert all(d['source_hashes'] == docs[0]['source_hashes'] for d in docs)
    arrays = [np.load(p/'frame_metrics.npz') for p in paths]
    rows, lookup, comparisons, noise = [], {}, [], []
    rng = np.random.default_rng(720261113)
    for row in docs[0]['results']:
        key = f"{row['profile']}_{row['snr_db']}_{row['method']}"
        values = np.concatenate([a[key] for a in arrays], axis=1)
        lookup[key] = values
        result = dict(profile=row['profile'], snr_db=row['snr_db'], method=row['method'],
                      frames=values.shape[1], nmse_db=float(10*np.log10(values[0].sum()/values[1].sum())),
                      ber=float(values[2].sum()/(96*values.shape[1])))
        seconds = [next(r['total_seconds'] for r in d['results'] if r['profile']==row['profile']
                        and r['snr_db']==row['snr_db'] and r['method']==row['method']) for d in docs]
        if seconds[0] is not None:
            result['median_seconds_per_frame'] = float(np.median([s/d['frames'] for s,d in zip(seconds, docs)]))
        rows.append(result)
    for row in docs[0]['noise']:
        key = f"{row['profile']}_{row['snr_db']}_{row['source']}_noise"
        values = np.concatenate([a[key] for a in arrays])
        v = 10**(-row['snr_db']/10)
        noise.append(dict(profile=row['profile'], snr_db=row['snr_db'], source=row['source'],
                          mean_ratio=float(values.mean()/v), relative_std=float(values.std()/v)))
    for profile in ['exp2','exp8','sparse_integer','fractional_sparse','close_fractional']:
        for snr in [0,10,20]:
            for candidate, reference in pairs:
                ca = lookup[f'{profile}_{snr}_{candidate}']
                re = lookup[f'{profile}_{snr}_{reference}']
                samples = []
                ber_samples = []
                for _ in range(20):
                    indices = rng.integers(ca.shape[1], size=(20, ca.shape[1]))
                    samples.extend(10*np.log10(re[0,indices].sum(axis=1)/ca[0,indices].sum(axis=1)))
                    ber_samples.extend((re[2,indices]-ca[2,indices]).mean(axis=1)/96)
                comparisons.append(dict(profile=profile,snr_db=snr,candidate=candidate,reference=reference,
                                        gain_db=float(10*np.log10(re[0].sum()/ca[0].sum())),
                                        ci95=np.quantile(samples,[.025,.975]).tolist(),
                                        ber_reduction=float((re[2]-ca[2]).mean()/96),
                                        ber_reduction_ci95=np.quantile(ber_samples,[.025,.975]).tolist()))
    out = Path('results')/output
    out.mkdir(exist_ok=True)
    document = dict(status='exploratory_paired_bootstrap_no_multiplicity_adjustment',
                    seeds=[d['seed'] for d in docs], frames_per_cell=sum(d['frames'] for d in docs),
                    bootstrap_replicates=400, results=rows, noise=noise, comparisons=comparisons)
    (out/'summary.json').write_text(json.dumps(document,indent=2),encoding='utf-8')
    lines = ['Positive gain favors candidate. Exploratory paired group bootstrap, 400 replicates.']
    for row in comparisons:
        lines.append(f"{row['profile']} {row['snr_db']}dB: {row['candidate']} vs {row['reference']} "
                     f"gain={row['gain_db']:.3f} dB CI95={row['ci95']}")
    (out/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(f'{prefix}: {len(rows)} pooled rows, {len(comparisons)} paired comparisons')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['pooling','offgrid'], required=True)
    args = p.parse_args()
    if args.mode == 'pooling':
        summarize('stage3_pool', 'stage3_pool_summary', [
            ('eb_residual_16','eb_residual_1'),
            ('eb_extra_noise_probes_64','eb_residual_16'),
        ])
    else:
        summarize('stage3_offgrid', 'stage3_offgrid_summary', [
            ('refined_residual16','eb_residual_16'),
            ('refined_residual16','cyclic_extended_eb_residual16'),
            ('refined_residual16','grid_residual16'),
            ('refined_residual16','refined_residual1'),
            ('refined_true_noise_ref','refined_residual16'),
        ])
