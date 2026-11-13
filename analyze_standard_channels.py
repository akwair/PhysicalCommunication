"""Aggregate EPA/EVA paired experiments without selecting cells or seeds."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from standard_channel_study import METHODS, SOURCES, SNR_DB, standard_profiles
from waveform_model import projection_noise_bias
from pilot_ofdm import PILOTS


def paired_interval(candidate, reference, group_counts, rng, replicates=2000):
    """Paired bootstrap, stratified by independently generated seed batches."""
    gain, ber = [], []
    offsets = np.cumsum([0]+group_counts[:-1]).tolist()
    for start in range(0, replicates, 100):
        batch = min(100, replicates-start)
        indices = np.concatenate([offset+rng.integers(n, size=(batch, n))
                                  for offset, n in zip(offsets, group_counts)], axis=1)
        gain.extend(10*np.log10(reference[0, indices].sum(axis=1)/candidate[0, indices].sum(axis=1)))
        ber.extend((reference[2, indices]-candidate[2, indices]).mean(axis=1)/96)
    return dict(gain_db=float(10*np.log10(reference[0].sum()/candidate[0].sum())),
                ci95=np.quantile(gain, [.025, .975]).tolist(),
                ber_reduction=float((reference[2]-candidate[2]).mean()/96),
                ber_ci95=np.quantile(ber, [.025, .975]).tolist())


def run(paths, out):
    docs = [json.loads((p/'results.json').read_text()) for p in paths]
    if len({d['seed'] for d in docs}) != len(docs) or not all(d['complete'] for d in docs):
        raise ValueError('Need distinct seeds and completed runs')
    if not all(d['source_hashes'] == docs[0]['source_hashes'] for d in docs):
        raise ValueError('Experiment sources differ between seeds')
    for name, digest in docs[0]['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Executed source changed: '+name)
    arrays = [np.load(p/'frame_metrics.npz') for p in paths]
    rng = np.random.default_rng(1620261113)
    groups = [d['groups_per_cell'] for d in docs]
    rows, noises, comparisons, diagnostics = [], [], [], []
    for profile, (delays, powers) in standard_profiles().items():
        bias = projection_noise_bias(delays, powers, PILOTS, 12)
        for snr in SNR_DB:
            prefix = f'{profile}_{snr}_'
            values = {method: np.concatenate([a[prefix+method] for a in arrays], axis=1) for method in METHODS}
            for method, data in values.items():
                if data.shape != (3, sum(groups)) or not np.all(np.isfinite(data)):
                    raise ValueError('Incomplete/nonfinite metric arrays')
                rows.append(dict(profile=profile, snr_db=snr, method=method, groups=sum(groups),
                                 nmse_db=float(10*np.log10(data[0].sum()/data[1].sum())),
                                 ber=float(data[2].sum()/(96*sum(groups)))))
            for source in SOURCES:
                ratio = np.concatenate([a[prefix+source+'_noise'] for a in arrays])*10**(snr/10)
                expected = 1+bias*10**(snr/10) if source == 'residual16' else 1.0
                noises.append(dict(profile=profile, snr_db=snr, source=source,
                                   mean_ratio=float(ratio.mean()), std_ratio=float(ratio.std(ddof=1)),
                                   se_mean_ratio=float(ratio.std(ddof=1)/np.sqrt(len(ratio))), expected_ratio=expected))
                calls = [c for d in docs for cell in d['diagnostics']
                         if (cell['profile'], cell['snr_db'], cell['source']) == (profile, snr, source)
                         for c in cell['calls']]
                diagnostics.append(dict(profile=profile, snr_db=snr, source=source, calls=len(calls),
                    selected_failure=sum(not c['refinement_success'][c['selected_order']] for c in calls),
                    any_candidate_failure=sum(not all(c['refinement_success']) for c in calls),
                    selected_at_order_cap=sum(c['selected_order'] == c['order_cap'] for c in calls),
                    mean_selected_order=float(np.mean([c['selected_order'] for c in calls]))))
            for method in ('eb', 'refined'):
                for candidate, reference in [(method+'_cp16', method+'_residual16'),
                                              (method+'_known_noise', method+'_cp16')]:
                    comparisons.append(dict(profile=profile, snr_db=snr, candidate=candidate, reference=reference,
                        **paired_interval(values[candidate], values[reference], groups, rng)))
    output = dict(groups_per_cell=sum(groups), seeds=[d['seed'] for d in docs],
                  total_groups=6*sum(groups), bootstrap_replicates=2000,
                  bootstrap='paired group resampling within seed; pointwise intervals',
                  results=rows, noise=noises, comparisons=comparisons, diagnostics=diagnostics,
                  waveform_audit=docs[0]['waveform_audit'],
                  per_seed_results=[dict(seed=d['seed'], results=d['results']) for d in docs],
                  source_hashes=docs[0]['source_hashes'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    for name, data in [('metrics', rows), ('noise', noises), ('comparisons', comparisons), ('diagnostics', diagnostics)]:
        with (out/(name+'.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for row in comparisons:
        if row['candidate'] == 'refined_cp16':
            print(row)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', type=Path, default=[Path(f'results/standard_channel{i}') for i in (1, 2, 3)])
    parser.add_argument('--out', type=Path, default=Path('results/standard_channel_summary'))
    args = parser.parse_args()
    run(args.inputs, args.out)
