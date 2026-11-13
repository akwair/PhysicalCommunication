"""Stage 2: exploratory unknown-noise baselines; no novelty claim.

All practical methods receive pilots only, plus a declared integer-delay support
bound and nominal ensemble channel power of one. No actual SNR/PDP is supplied.
Fractional paths use a continuous-delay frequency-response stress model, not a
validated time-domain fractional-delay waveform simulator.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import roots_legendre, softmax
from pilot_ofdm import cn, N, PILOTS, DATA

M = len(PILOTS)


def fourier(delays):
    return np.exp(-2j*np.pi*np.outer(np.arange(N), delays)/N)


def statistics(y, length):
    if not 0 < length < M:
        raise ValueError('Residual noise estimation requires 0 < length < pilots.')
    basis = fourier(np.arange(length))
    a = basis[PILOTS]
    z = y @ a.conj()/M
    residual = y-z @ a.T
    rss = np.sum(abs(residual)**2, axis=1)
    return z, rss, basis


def priors(length):
    p = np.exp(-np.arange(length)[None, :]/np.array([.5, 1, 2, 4, 8, 16])[:, None])
    p /= p.sum(axis=1, keepdims=True)
    return np.vstack([p, np.ones(length)/length])


def mix_estimate(z, variance, bank):
    variance = np.broadcast_to(np.asarray(variance), len(z))
    total = bank[None, :, :] + variance[:, None, None]/M
    logs = -np.sum(np.log(total)+abs(z[:, None, :])**2/total, axis=2)
    weight = softmax(logs, axis=1)
    return z*np.sum(weight[:, :, None]*bank[None, :, :]/total, axis=1)


def integrated_mix(z, rss, dof, bank, nodes=128, cut=False):
    """p(v) proportional 1/v. p(v|residual)=IG(dof,rss).

    Integrate jointly over v and the discrete PDP model. The subspace likelihood
    updates the residual posterior; residual evidence is not counted twice.
    Gauss-Legendre quadrature in log(v/rss), with data-scaled integration bounds.
    The Jacobian is included: the residual log density is -dof*log(v)-rss/v.
    """
    x, w = roots_legendre(nodes)
    logw = np.log(w/w.sum())
    output = np.empty_like(z)
    for start in range(0, len(z), 256):
        zz = z[start:start+256]
        rr = np.maximum(rss[start:start+256], 1e-300)
        lower = -np.log(dof+z.shape[1])-8
        upper = np.log(np.maximum(1, M*np.sum(abs(zz)**2, axis=1)/rr))+10
        t = lower+(x[None, :]+1)*(upper[:, None]-lower)/2
        variances = rr[:, None]*np.exp(t)
        total = bank[None, :, None, :] + variances[:, None, :, None]/M
        logs = -np.sum(np.log(total)+abs(zz[:, None, None, :])**2/total, axis=-1)
        residual_logs = logw[None, :]-dof*t-np.exp(-t)
        if cut:
            # Modular diagnostic: do not let a potentially wrong channel prior
            # update noise inferred from the residual. This is NOT full Bayes.
            weight = softmax(logs, axis=1)*softmax(residual_logs, axis=1)[:, None, :]
        else:
            logs += residual_logs[:, None, :]
            weight = softmax(logs.reshape(len(zz), -1), axis=1).reshape(logs.shape)
        shrink = np.sum(weight[..., None]*bank[None, :, None, :]/total, axis=(1, 2))
        output[start:start+len(zz)] = zz*shrink
    return output


def soft_threshold(z, threshold):
    return z*np.maximum(0, 1-np.asarray(threshold)/np.maximum(abs(z), 1e-300))


def sure_risk(z, vz, threshold):
    """Complex SURE for a FIXED soft threshold; selection adds optimism."""
    magnitude = np.maximum(abs(z), 1e-300)
    eta = soft_threshold(z, threshold)
    divergence = np.where(magnitude > threshold, 2-threshold/magnitude, 0)
    return np.sum(abs(eta-z)**2 + vz*divergence-vz, axis=-1)


def sure_selected(z, variance):
    vz = variance[:, None]/M
    risks, estimates = [], []
    for factor in [0, .5, 1, 1.5, 2, 2.5, 3]:
        threshold = factor*np.sqrt(vz)
        estimates.append(soft_threshold(z, threshold))
        risks.append(sure_risk(z, vz, threshold))
    index = np.argmin(risks, axis=0)
    return np.asarray(estimates)[index, np.arange(len(z))]


def diagonal_em(z, rss, initial_variance, max_iter=200):
    """Standard single-snapshot ARD EM specialized to orthogonal pilot columns.

    Independent tap prior variances and noise variance updated by EM. This is a
    generic baseline, not a reproduction of a particular published OFDM method.
    """
    v = np.maximum(initial_variance, 1e-12).copy()
    gamma = np.maximum(abs(z)**2, 1e-8)
    for _ in range(max_iter):
        total = gamma+v[:, None]/M
        mean = z*gamma/total
        cov = gamma*(v[:, None]/M)/total
        new_gamma = np.maximum(abs(mean)**2+cov, 1e-12)
        new_v = np.maximum((rss+M*np.sum(abs(z-mean)**2+cov, axis=1))/M, 1e-12)
        change = np.max(abs(new_v-v)/v)
        change = max(change, np.max(abs(new_gamma-gamma)/(gamma+1e-8)))
        gamma, v = new_gamma, new_v
        if change < 1e-6:
            break
    return z*gamma/(gamma+v[:, None]/M), v


def practical(y, length, nodes):
    z, rss, basis = statistics(y, length)
    v = np.maximum(rss/(M-length), 1e-12)
    bank = priors(length)
    outputs = {}
    timings = {}
    methods = {
        'delay_ls': lambda: z,
        'uniform_plugin': lambda: z/(1+length*v[:, None]/M),
        'mixture_plugin': lambda: mix_estimate(z, v, bank),
        'mixture_integrated': lambda: integrated_mix(z, rss, M-length, bank, nodes),
        'mixture_cut': lambda: integrated_mix(z, rss, M-length, bank, nodes, cut=True),
        'sure_soft_plugin': lambda: sure_selected(z, v),
        'diagonal_eb_plugin': lambda: z*np.maximum(0, 1-(v[:, None]/M)/np.maximum(abs(z)**2, 1e-300)),
        'diagonal_em_joint': lambda: diagonal_em(z, rss, v)[0],
    }
    for name, operation in methods.items():
        start = time.perf_counter()
        outputs[name] = operation() @ basis.T
        timings[name] = time.perf_counter()-start
    return outputs, v, timings


def profiles():
    short = np.exp(-np.arange(12)/2)
    broad = np.exp(-np.arange(12)/8)
    return {
        'exp2': (np.arange(12), short/short.sum()),
        'exp8': (np.arange(12), broad/broad.sum()),
        'sparse_integer': (np.array([0, 7, 11]), np.array([.5, .3, .2])),
        'fractional_stress': (np.array([0, 7.35, 11.2]), np.array([.5, .3, .2])),
    }


def run(frames, seed, nodes, out):
    rng = np.random.default_rng(seed)
    rows, raw, noise_rows = [], {}, []
    for name, (delays, power) in profiles().items():
        f = fourier(delays)
        cov = (f*power) @ f.conj().T
        for snr in [0, 10, 20]:
            variance = 10**(-snr/10)
            truth = (cn(rng, (frames, len(power)))*np.sqrt(power)) @ f.T
            y = truth[:, PILOTS]+np.sqrt(variance)*cn(rng, (frames, M))
            bits = rng.integers(0, 2, (frames, len(DATA), 2))
            x = ((1-2*bits[:, :, 0])+1j*(1-2*bits[:, :, 1]))/np.sqrt(2)
            yd = truth[:, DATA]*x+np.sqrt(variance)*cn(rng, x.shape)
            for length in [12, 15]:
                output, vhat, timing = practical(y, length, nodes)
                z, _, basis = statistics(y, length)
                bank = priors(length)
                output['mixture_known_noise_ref'] = mix_estimate(z, variance, bank) @ basis.T
                output['uniform_known_noise_ref'] = z/(1+length*variance/M) @ basis.T
                c = cov[np.ix_(PILOTS, PILOTS)]+variance*np.eye(M)
                gain = np.linalg.solve(c, cov[PILOTS]).conj().T
                output['oracle_lmmse_ref'] = y @ gain.T
                noise_rows.append(dict(profile=name, snr_db=snr, length=length,
                                       mean_ratio=float(np.mean(vhat)/variance),
                                       relative_std=float(np.std(vhat)/variance)))
                raw[f'{name}_{snr}_{length}_noise'] = vhat
                for method, predicted in output.items():
                    error = np.mean(abs(predicted-truth)**2, axis=1)
                    energy = np.mean(abs(truth)**2, axis=1)
                    detected = yd*predicted[:, DATA].conj()
                    decoded = np.stack([detected.real < 0, detected.imag < 0], axis=-1)
                    bit_errors = np.sum(decoded != bits, axis=(1, 2))
                    nmse = error.sum()/energy.sum()
                    rows.append(dict(profile=name, snr_db=snr, length=length, method=method,
                                     nmse_db=float(10*np.log10(nmse)), ber=float(bit_errors.sum()/bits.size),
                                     bit_errors=int(bit_errors.sum()), bits=int(bits.size),
                                     total_batch_seconds=timing.get(method), frames=frames))
                    raw[f'{name}_{snr}_{length}_{method}'] = np.stack([error, energy, bit_errors])
            print(f'{name}: {snr} dB done', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'frame_metrics.npz', **raw)
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    result = dict(status='exploratory_not_confirmatory', seed=seed, frames=frames, nodes=nodes,
                  versions=versions, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(Path('pilot_ofdm.py').read_bytes()).hexdigest(),
                  noise=noise_rows, results=rows)
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=2000)
    p.add_argument('--seed', type=int, default=320261113)
    p.add_argument('--nodes', type=int, default=128)
    p.add_argument('--out', type=Path, default=Path('results/noise_pilot'))
    args = p.parse_args()
    if args.frames < 100 or args.nodes < 8:
        p.error('frames >= 100 and nodes >= 8 required')
    run(args.frames, args.seed, args.nodes, args.out)
