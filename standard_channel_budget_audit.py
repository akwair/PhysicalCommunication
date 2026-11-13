"""Replay all nonconverged EPA/EVA fits at a larger optimizer budget."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import offgrid_baseline
from pilot_ofdm import PILOTS
from stage3_study import FREQ, metrics
from standard_channel_study import json_value


def run():
    out = Path('results/standard_channel_budget')
    if (out/'summary.json').exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    original = offgrid_baseline.least_squares
    def higher_budget(*args, **kwargs):
        kwargs['max_nfev'] = 300
        return original(*args, **kwargs)
    batches, replayed = [], []
    started = perf_counter()
    for rep in (1, 2, 3):
        path = Path(f'results/standard_channel{rep}')
        doc = json.loads((path/'results.json').read_text())
        for name, digest in doc['source_hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        with np.load(path/'frame_metrics.npz') as saved:
            raw = {key: saved[key].copy() for key in saved.files}
        with np.load(path/'capped_inputs.npz') as inputs:
            with patch.object(offgrid_baseline, 'least_squares', higher_budget):
                for cell in doc['diagnostics']:
                    prefix = f"{cell['profile']}_{cell['snr_db']}_"
                    source = cell['source']
                    for index, call in enumerate(cell['calls']):
                        if all(call['refinement_success']):
                            continue
                        key = prefix+str(index)
                        fit = offgrid_baseline.estimate_offgrid(inputs[key+'_pilot'], FREQ[PILOTS], FREQ,
                            float(inputs[key+'_'+source+'_noise']), max_delay=12, grid_step=.25,
                            max_paths=6, refine=True)
                        raw[prefix+'refined_'+source][:, index] = metrics(fit['response'][None],
                            inputs[key+'_truth'][None], inputs[key+'_bits'][None], inputs[key+'_received'][None])[:, 0]
                        raw[prefix+'refined_'+source+'_orders'][index] = fit['selected_order']
                        replayed.append(dict(seed=doc['seed'], profile=cell['profile'], snr_db=cell['snr_db'],
                            source=source, index=index, selected_order=fit['selected_order'],
                            refinement_success=fit['refinement_success'], nfev_history=fit['nfev_history']))
        np.savez_compressed(out/f'frame_metrics{rep}.npz', **raw)
        batches.append(raw)
        print(f'replicate {rep} replayed; total {len(replayed)} calls; {perf_counter()-started:.1f}s', flush=True)
    baseline = json.loads(Path('results/standard_channel_summary/summary.json').read_text())
    rows = []
    for row in baseline['results']:
        if not row['method'].startswith('refined_'):
            continue
        key = f"{row['profile']}_{row['snr_db']}_{row['method']}"
        a = np.concatenate([batch[key] for batch in batches], axis=1)
        nmse = float(10*np.log10(a[0].sum()/a[1].sum()))
        rows.append(dict(profile=row['profile'], snr_db=row['snr_db'], method=row['method'],
                         nmse_db=nmse, ber=float(a[2].sum()/(96*a.shape[1])),
                         nmse_change_db=nmse-row['nmse_db']))
    gains = []
    lookup = {(r['profile'], r['snr_db'], r['method']):r for r in rows}
    for r in baseline['comparisons']:
        if r['candidate'] != 'refined_cp16':
            continue
        key = (r['profile'], r['snr_db'])
        gain = lookup[key+('refined_residual16',)]['nmse_db'] - lookup[key+('refined_cp16',)]['nmse_db']
        gains.append(dict(profile=r['profile'], snr_db=r['snr_db'], gain_db=gain,
                          baseline_gain_db=r['gain_db'], gain_change_db=gain-r['gain_db']))
    report = dict(budget=300, replayed_calls=len(replayed),
        remaining_any_candidate_failure=sum(not all(r['refinement_success']) for r in replayed),
        remaining_selected_failure=sum(not r['refinement_success'][r['selected_order']] for r in replayed),
        results=rows, gains=gains, diagnostics=replayed, elapsed_seconds=perf_counter()-started,
        source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('standard_channel_budget_audit.py', 'standard_channel_budget_protocol.txt', 'offgrid_baseline.py')})
    (out/'summary.json').write_text(json.dumps(report, default=json_value, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('diagnostics','source_hashes','results')}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
