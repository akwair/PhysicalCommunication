"""Stage-3 exploratory controls with explicitly signed OFDM frequencies.

Calibration groups: T=16 independent fading blocks with identical path geometry
and constant noise variance. Pool only noise statistics, not channel amplitudes.
External noise probes are extra-resource controls, not free observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import scipy
from pilot_ofdm import cn, N, PILOTS, DATA
from noise_study import priors, mix_estimate

FREQ = np.fft.fftfreq(N)
L = 12
M = len(PILOTS)
T = 16


def response_matrix(delays):
    return np.exp(-2j*np.pi*np.outer(FREQ, delays))


def stage3_profiles():
    short = np.exp(-np.arange(12)/2)
    broad = np.exp(-np.arange(12)/8)
    return {
        'exp2': (np.arange(12), short/short.sum()),
        'exp8': (np.arange(12), broad/broad.sum()),
        'sparse_integer': (np.array([0, 7, 11]), np.array([.5, .3, .2])),
        'fractional_sparse': (np.array([0, 7.35, 11.2]), np.array([.5, .3, .2])),
        'close_fractional': (np.array([0, 3.1, 3.45, 10.7]), np.array([.4, .2, .2, .2])),
    }


def projected_statistics(y):
    basis = response_matrix(np.arange(L))
    a = basis[PILOTS]
    z = y @ a.conj()/M
    residual = y-z @ a.T
    return z, np.sum(abs(residual)**2, axis=-1)/(M-L), basis


def pool_noise(vhat, blocks):
    if vhat.ndim != 2 or blocks < 1 or blocks > vhat.shape[1]:
        raise ValueError('Noise pool needs independent groups and a valid block count.')
    return np.mean(vhat[:, :blocks], axis=1)


def generate(rng, frames, delays, power, variance):
    f = response_matrix(delays)
    gains = cn(rng, (frames, T, len(delays)))*np.sqrt(power)
    channels = gains @ f.T
    yp = channels[:, :, PILOTS]+np.sqrt(variance)*cn(rng, (frames, T, M))
    bits = rng.integers(0, 2, (frames, len(DATA), 2))
    symbols = ((1-2*bits[:, :, 0])+1j*(1-2*bits[:, :, 1]))/np.sqrt(2)
    yd = channels[:, 0, DATA]*symbols+np.sqrt(variance)*cn(rng, symbols.shape)
    probes = np.sqrt(variance)*cn(rng, (frames, T*(M-L)))
    return channels[:, 0], yp, bits, yd, np.mean(abs(probes)**2, axis=1)


def metrics(estimate, truth, bits, received):
    error = np.mean(abs(estimate-truth)**2, axis=1)
    energy = np.mean(abs(truth)**2, axis=1)
    eq = received*estimate[:, DATA].conj()
    decoded = np.stack([eq.real < 0, eq.imag < 0], axis=-1)
    errors = np.sum(decoded != bits, axis=(1, 2))
    return np.stack([error, energy, errors])


def run(args):
    rng = np.random.default_rng(args.seed)
    raw, rows, noise = {}, [], []
    for profile, (delays, power) in stage3_profiles().items():
        for snr in [0, 10, 20]:
            v = 10**(-snr/10)
            truth, yp, bits, yd, probe = generate(rng, args.frames, delays, power, v)
            zall, vhat, basis = projected_statistics(yp)
            z = zall[:, 0]
            sources = {
                'residual_1': pool_noise(vhat, 1),
                'residual_4': pool_noise(vhat, 4),
                'residual_16': pool_noise(vhat, 16),
                'extra_noise_probes_64': probe,
                'true_noise_ref': np.full(args.frames, v),
            }
            methods = {'delay_ls': z @ basis.T}
            seconds = {}
            orders = {}
            # Cheap leakage-aware control: circularly extended integer window.
            # Fifteen distinct pilot-DFT columns [-2,...,12], one residual DOF.
            # It is an adapted conventional DFT window, not a new algorithm.
            ext_basis = response_matrix(np.arange(-2, 13))
            ext_a = ext_basis[PILOTS]
            ext_z = yp @ ext_a.conj()/M
            ext_residual = yp-ext_z @ ext_a.T
            ext_v = np.mean(np.sum(abs(ext_residual)**2, axis=-1), axis=1)
            methods['cyclic_extended_ls'] = ext_z[:,0] @ ext_basis.T
            methods['cyclic_extended_eb_residual16'] = (ext_z[:,0]*np.maximum(
                0, 1-(ext_v[:,None]/M)/np.maximum(abs(ext_z[:,0])**2,1e-300))) @ ext_basis.T
            for source, estimate_v in sources.items():
                vv = np.maximum(estimate_v, 1e-12)
                noise.append(dict(profile=profile, snr_db=snr, source=source,
                                  mean_ratio=float(estimate_v.mean()/v),
                                  relative_std=float(estimate_v.std()/v)))
                raw[f'{profile}_{snr}_{source}_noise'] = estimate_v
                methods[f'uniform_{source}'] = z/(1+L*vv[:, None]/M) @ basis.T
                methods[f'eb_{source}'] = (z*np.maximum(0, 1-(vv[:, None]/M)/np.maximum(abs(z)**2, 1e-300))) @ basis.T
                methods[f'mixture_{source}'] = mix_estimate(z, vv, priors(L)) @ basis.T
            if args.offgrid:
                from offgrid_baseline import estimate_offgrid
                configurations = [
                    ('grid_residual16', 'residual_16', False),
                    ('refined_residual1', 'residual_1', True),
                    ('refined_residual16', 'residual_16', True),
                    ('refined_true_noise_ref', 'true_noise_ref', True),
                ]
                for name, source, refine in configurations:
                    begin = time.perf_counter()
                    estimated, k = [], []
                    for i in range(args.frames):
                        fit = estimate_offgrid(yp[i, 0], FREQ[PILOTS], FREQ,
                                               max(sources[source][i], 1e-12),
                                               max_delay=12., grid_step=.25, max_paths=6,
                                               refine=refine)
                        estimated.append(fit['response'])
                        k.append(fit['selected_order'])
                    methods[name] = np.asarray(estimated)
                    seconds[name] = time.perf_counter()-begin
                    orders[name] = np.asarray(k)
            f = response_matrix(delays)
            covariance = (f*power) @ f.conj().T
            c = covariance[np.ix_(PILOTS, PILOTS)]+v*np.eye(M)
            gain = np.linalg.solve(c, covariance[PILOTS]).conj().T
            methods['oracle_lmmse_ref'] = yp[:, 0] @ gain.T
            for name, estimate in methods.items():
                values = metrics(estimate, truth, bits, yd)
                raw[f'{profile}_{snr}_{name}'] = values
                row = dict(profile=profile, snr_db=snr, method=name, frames=args.frames,
                           nmse_db=float(10*np.log10(values[0].sum()/values[1].sum())),
                           ber=float(values[2].sum()/(96*args.frames)),
                           total_seconds=seconds.get(name))
                if name in orders:
                    raw[f'{profile}_{snr}_{name}_orders'] = orders[name]
                    row['mean_selected_order'] = float(orders[name].mean())
                rows.append(row)
            print(f'{profile} {snr} dB completed', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'frame_metrics.npz', **raw)
    names = ['stage3_study.py', 'noise_study.py', 'pilot_ofdm.py']
    if args.offgrid:
        names.append('offgrid_baseline.py')
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    document = dict(status='exploratory_signed_frequency_model', seed=args.seed, frames=args.frames,
                    offgrid=args.offgrid, numpy=np.__version__, scipy=scipy.__version__,
                    source_hashes=hashes, results=rows, noise=noise,
                    assumptions={'calibration_blocks':16, 'noise_constant_in_group':True,
                                 'independent_channel_gains_in_group':True,
                                 'uses_joint_multiblock_channel_estimation':False,
                                 'extra_noise_probes_are_additional_resources':True})
    (args.out/'results.json').write_text(json.dumps(document, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=520261113)
    parser.add_argument('--offgrid', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('results/stage3_pooling'))
    args = parser.parse_args()
    if args.frames < 100:
        parser.error('At least 100 groups required')
    run(args)
