"""Fixed-receiver validation with 3GPP EPA/EVA power-delay profiles.

See standard_channel_protocol.txt for the frozen scope and physical units.
This uses standard PDPs, not standardized LTE Doppler or an LTE waveform.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from cp_confirmation import generate_cp
from offgrid_baseline import estimate_offgrid
from pilot_ofdm import cn, N, PILOTS
from stage3_study import FREQ, L, M, projected_statistics, metrics, response_matrix
from waveform_model import modulate, receive_ofdm, projection_noise_bias

SAMPLE_NS = 250.0
SNR_DB = (0, 10, 20)
SOURCES = ('residual16', 'cp16', 'known_noise')
METHODS = ('delay_ls',) + tuple(f'{method}_{source}' for method in ('eb', 'refined') for source in SOURCES)
PROFILE_TABLE = {
    'EPA': ([0, 30, 70, 90, 110, 190, 410], [0, -1, -2, -3, -8, -17.2, -20.8]),
    'EVA': ([0, 30, 150, 310, 370, 710, 1090, 1730, 2510], [0, -1.5, -1.4, -3.6, -.6, -9.1, -7, -12, -16.9]),
}
HASH_FILES = ('standard_channel_study.py', 'standard_channel_protocol.txt',
              'cp_confirmation.py', 'offgrid_baseline.py', 'stage3_study.py',
              'waveform_model.py', 'pilot_ofdm.py', 'noise_study.py')


def standard_profiles():
    result = {}
    for name, (ns, db) in PROFILE_TABLE.items():
        powers = 10**(np.asarray(db, float)/10)
        result[name] = (np.asarray(ns, float)/SAMPLE_NS, powers/powers.sum())
    return result


def waveform_audit(seed=14731):
    rng = np.random.default_rng(seed)
    symbols = cn(rng, (3, N))
    oversample = 25
    tx = modulate(symbols, cp_samples=16, oversample=oversample)
    report = {}
    for name, (delays, powers) in standard_profiles().items():
        gains = cn(rng, (len(delays),))*np.sqrt(powers)
        shifts = np.rint(delays*oversample).astype(int)
        rx = np.zeros(len(tx)+max(shifts), complex)
        for shift, gain in zip(shifts, gains):
            rx[shift:shift+len(tx)] += gain*tx
        left = (80+np.arange(12, 16))*oversample
        actual = receive_ofdm(symbols, delays, gains, cp_samples=16, oversample=oversample)
        expected = symbols*(response_matrix(delays) @ gains)
        report[name] = dict(
            max_cp_signal_difference=float(np.max(abs(rx[left]-rx[left+N*oversample]))),
            max_frequency_response_error=float(np.max(abs(actual-expected))))
    return report


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run(args):
    if args.frames < 1:
        raise ValueError('frames must be positive')
    if (args.out/'results.json').exists():
        raise FileExistsError('Choose a new output directory; existing experiment will not be overwritten.')
    args.out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rng = np.random.default_rng(args.seed)
    raw, rows, diagnostics, capped_inputs = {}, [], [], {}
    record = dict(seed=args.seed, groups_per_cell=args.frames, sample_interval_ns=SAMPLE_NS,
                  source_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in HASH_FILES},
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  profile_table_ns_db=PROFILE_TABLE, waveform_audit=waveform_audit())
    for profile, (delays, powers) in standard_profiles().items():
        bias = projection_noise_bias(delays, powers, PILOTS, L)
        for snr in SNR_DB:
            prefix = f'{profile}_{snr}_'
            variance = 10**(-snr/10)
            truth, yp, bits, yd, cp = generate_cp(rng, args.frames, delays, powers, variance)
            z, residual, basis = projected_statistics(yp)
            sources = dict(residual16=residual.mean(axis=1), cp16=cp,
                           known_noise=np.full(args.frames, variance))
            raw[prefix+'delay_ls'] = metrics(z[:, 0] @ basis.T, truth, bits, yd)
            for source, noise in sources.items():
                raw[prefix+source+'_noise'] = noise
                estimate = (z[:, 0]*np.maximum(0, 1-noise[:, None]/M/
                             np.maximum(abs(z[:, 0])**2, 1e-300))) @ basis.T
                raw[prefix+'eb_'+source] = metrics(estimate, truth, bits, yd)
                estimates, orders, calls = [], [], []
                for i in range(args.frames):
                    fit = estimate_offgrid(yp[i, 0], FREQ[PILOTS], FREQ, max(noise[i], 1e-12),
                                           max_delay=12, grid_step=.25, max_paths=6, refine=True)
                    estimates.append(fit['response'])
                    orders.append(fit['selected_order'])
                    calls.append({k: val for k, val in fit.items() if k not in ('response', 'delays', 'gains')})
                    if not all(fit['refinement_success']):
                        key = prefix+str(i)
                        capped_inputs[key+'_pilot'] = yp[i, 0]
                        capped_inputs[key+'_truth'] = truth[i]
                        capped_inputs[key+'_bits'] = bits[i]
                        capped_inputs[key+'_received'] = yd[i]
                        capped_inputs[key+'_'+source+'_noise'] = noise[i]
                raw[prefix+'refined_'+source] = metrics(np.asarray(estimates), truth, bits, yd)
                raw[prefix+'refined_'+source+'_orders'] = np.asarray(orders)
                diagnostics.append(dict(profile=profile, snr_db=snr, source=source, calls=calls))
            for method in METHODS:
                values = raw[prefix+method]
                rows.append(dict(profile=profile, snr_db=snr, method=method,
                                 nmse_db=float(10*np.log10(values[0].sum()/values[1].sum())),
                                 ber=float(values[2].sum()/(96*args.frames))))
            record.update(results=rows, diagnostics=diagnostics, complete=False,
                          elapsed_seconds=perf_counter()-started)
            (args.out/'checkpoint.json').write_text(json.dumps(record, default=json_value), encoding='utf-8')
            np.savez_compressed(args.out/'frame_metrics.npz', **raw)
            print(f'{profile} {snr} dB complete; analytic residual ratio {1+bias/variance:.4f}; '
                  f'elapsed {perf_counter()-started:.1f}s', flush=True)
    np.savez_compressed(args.out/'capped_inputs.npz', **capped_inputs)
    record.update(complete=True, elapsed_seconds=perf_counter()-started)
    (args.out/'results.json').write_text(json.dumps(record, default=json_value), encoding='utf-8')
    (args.out/'checkpoint.json').unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1520261113)
    parser.add_argument('--out', type=Path, default=Path('results/standard_channel1'))
    with threadpool_limits(limits=1):
        run(parser.parse_args())
