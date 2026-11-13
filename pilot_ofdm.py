"""Exploratory CPU benchmark. All estimators are established baselines, not novelty.

Run: python pilot_ofdm.py --frames 2000 --seed 20261113
Frequency-domain equivalent of synchronized block-static SISO CP-OFDM.
N=64, CP>=11, unit-energy symbols, 16 comb pilots, 12 integer-delay taps.
No Doppler, coding, RF impairments or fractional delays are simulated.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import softmax

N, L = 64, 12
PILOTS = np.arange(0, N, 4)
DATA = np.setdiff1d(np.arange(N), PILOTS)
F = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(L)) / N)
A = F[PILOTS]


def cn(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def pdp(tau):
    p = np.exp(-np.arange(L) / tau)
    return p / p.sum()


def filt(p, variance):
    cross = p[:, None] * A.conj().T
    cov = A @ cross + variance * np.eye(len(PILOTS))
    gain = np.linalg.solve(cov, cross.conj().T).conj().T
    return gain, cov


def estimate(y, p, variance):
    gain, _ = filt(p, variance)
    return (y @ gain.T) @ F.T


def mixture(y, variance):
    # Equal prior mixture of six fixed exponential PDPs; no fitted/test labels.
    estimates, logweights = [], []
    for tau in [0.5, 1, 2, 4, 8, 16]:
        gain, cov = filt(pdp(tau), variance)
        estimates.append((y @ gain.T) @ F.T)
        quadratic = np.sum(y.conj() * np.linalg.solve(cov, y.T).T, axis=1).real
        logweights.append(-np.linalg.slogdet(cov)[1] - quadratic)
    weights = softmax(np.array(logweights).T, axis=1)
    return np.einsum('bk,kbn->bn', weights, np.array(estimates))


def validate():
    rng = np.random.default_rng(731)
    h = cn(rng, (20000, L)) * np.sqrt(pdp(2))
    truth = h @ F.T
    recovered = (truth[:, PILOTS] @ np.linalg.pinv(A).T) @ F.T
    assert np.max(np.abs(recovered - truth)) < 1e-11
    v = 0.2
    y = truth[:, PILOTS] + np.sqrt(v) * cn(rng, (len(h), len(PILOTS)))
    gain, _ = filt(pdp(2), v)
    error_cov = np.diag(pdp(2)) - gain @ A @ np.diag(pdp(2))
    theory = np.trace(error_cov).real
    observed = np.mean(np.abs(estimate(y, pdp(2), v) - truth) ** 2)
    assert abs(observed / theory - 1) < 0.03, (observed, theory)
    ls = (y @ np.linalg.pinv(A).T) @ F.T
    ls_observed = np.mean(np.abs(ls - truth) ** 2)
    ls_theory = v * L / len(PILOTS)
    assert abs(ls_observed / ls_theory - 1) < 0.03
    return {'noiseless_max_error': float(np.max(np.abs(recovered-truth))),
            'lmmse_mse_observed': float(observed), 'lmmse_mse_theory': float(theory),
            'ls_mse_observed': float(ls_observed), 'ls_mse_theory': ls_theory}


def run(frames, seed, out):
    checks = validate()
    rng = np.random.default_rng(seed)
    sparse = np.zeros(L)
    sparse[[0, 7, 11]] = [0.5, 0.3, 0.2]
    profiles = {'matched_exp2': pdp(2), 'broad_exp8': pdp(8), 'off_family_sparse': sparse}
    rows, raw = [], {}
    for profile, power in profiles.items():
        for snr in [0, 10, 20]:
            variance = 10 ** (-snr / 10)
            truth = (cn(rng, (frames, L)) * np.sqrt(power)) @ F.T
            bits = rng.integers(0, 2, (frames, len(DATA), 2))
            symbols = ((1 - 2 * bits[:, :, 0]) + 1j * (1 - 2 * bits[:, :, 1])) / np.sqrt(2)
            y = truth[:, PILOTS] + np.sqrt(variance) * cn(rng, (frames, len(PILOTS)))
            received = truth[:, DATA] * symbols + np.sqrt(variance) * cn(rng, symbols.shape)
            for noise_factor in [1, 4]:
                assumed = variance * noise_factor
                estimates = {
                    'delay_ls': (y @ np.linalg.pinv(A).T) @ F.T,
                    'fixed_lmmse': estimate(y, pdp(2), assumed),
                    'uniform_lmmse': estimate(y, np.ones(L)/L, assumed),
                    'pdp_mixture': mixture(y, assumed),
                    'oracle_lmmse': estimate(y, power, variance),
                    'perfect_csi': truth,
                }
                for name, predicted in estimates.items():
                    error = np.mean(np.abs(predicted - truth) ** 2, axis=1)
                    energy = np.mean(np.abs(truth) ** 2, axis=1)
                    # QPSK decisions only need phase compensation, avoiding division near fades.
                    equalized = received * predicted[:, DATA].conj()
                    decisions = np.stack([equalized.real < 0, equalized.imag < 0], axis=-1)
                    errors = np.sum(decisions != bits, axis=(1, 2))
                    nmse = float(error.sum() / energy.sum())
                    rows.append(dict(profile=profile, snr_db=snr, assumed_noise_factor=noise_factor,
                                     method=name, nmse_linear=nmse,
                                     nmse_db=float(10*np.log10(nmse)) if nmse > 0 else None,
                                     ber=float(errors.sum()/bits.size), bit_errors=int(errors.sum()),
                                     bits=int(bits.size), frames=frames))
                    raw[f'{profile}_{snr}_{noise_factor}_{name}'] = np.stack([error, energy, errors])
            print(f'Completed {profile}, SNR={snr} dB', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'frame_metrics.npz', **raw)
    result = dict(status='exploratory_not_publication_evidence', seed=seed, frames_per_case=frames,
                  numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checks=checks, results=rows)
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--out', type=Path, default=Path('results/pilot'))
    args = parser.parse_args()
    if args.frames < 100:
        parser.error('Use at least 100 frames.')
    start = time.perf_counter()
    run(args.frames, args.seed, args.out)
    print(f'Elapsed: {time.perf_counter()-start:.2f} seconds')
