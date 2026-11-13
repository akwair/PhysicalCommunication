"""Prespecified geometry/window sensitivity of fixed-subspace noise calibration."""
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from waveform_model import delay_atoms

OUT = Path('results/operating_region')
POWERS = np.array([.5, .3, .2])
BLOCKS = (1, 4, 16)


def geometry(m, length, delta):
    pilots = np.arange(0, 64, 64 // m)
    basis = delay_atoms(np.arange(length))[pilots]
    u, _ = np.linalg.qr(basis, mode='reduced')
    q = np.eye(m) - u @ u.conj().T
    atoms = delay_atoms([0, 7 + delta, 11 + delta])[pilots]
    residual_atoms = q @ atoms
    s = (residual_atoms * POWERS) @ residual_atoms.conj().T
    return q, atoms, s


def prediction(m, length, delta, snr):
    _, _, s = geometry(m, length, delta)
    d = m - length
    v = 10. ** (-snr / 10)
    leakage = float(np.trace(s).real)
    # Exact integer-delay geometry is matched; remove floating-point residue.
    if delta == 0:
        assert leakage < 1e-24
        leakage = 0.
    tr_s2 = float(np.vdot(s, s).real)
    if delta == 0:
        tr_s2 = 0.
    bias = leakage / d
    a = (tr_s2 + 2 * v * leakage + v * v * d) / (d * d)
    return dict(m=m, length=length, delta=float(delta), snr_db=snr, d=d,
                variance=v, raw_leakage=leakage, trace_s_squared=tr_s2,
                bias=bias, relative_bias=bias/v, single_block_variance=a,
                crossover_blocks=a/bias**2 if bias > 0 else None,
                normalized_mse={str(t): (bias**2 + a/t)/v**2 for t in BLOCKS})


def simulate(m, length, delta, seed, groups=20000):
    q, atoms, _ = geometry(m, length, delta)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, groups, 1000):
        n = min(1000, groups-start)
        gains = (rng.standard_normal((n, 16, 3)) +
                 1j*rng.standard_normal((n, 16, 3))) * np.sqrt(POWERS/2)
        noise = (rng.standard_normal((n, 16, m)) +
                 1j*rng.standard_normal((n, 16, m))) * np.sqrt(.01/2)
        y = gains @ atoms.T + noise
        residual = y @ q.T
        vhat = np.sum(abs(residual)**2, axis=-1)/(m-length)
        estimates.append(np.stack([vhat[:, :t].mean(axis=1) for t in BLOCKS], axis=1))
    return np.concatenate(estimates)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    analytic = [prediction(m, length, delta/10, snr)
                for m in (16, 32) for delta in range(10)
                for length in range(12, 16) for snr in (10, 20)]
    cases = [(m, length, delta) for m in (16, 32)
             for delta in (0., .2, .5, .9) for length in (12, 15)]
    seeds = np.random.SeedSequence(1320261113).spawn(len(cases))
    arrays, checks = {}, []
    for case_id, ((m, length, delta), seed) in enumerate(zip(cases, seeds)):
        est = simulate(m, length, delta, seed)
        p = prediction(m, length, delta, 20)
        arrays[f'case_{case_id:02d}'] = est
        for j, t in enumerate(BLOCKS):
            values = est[:, j]
            squared_error = (values/.01 - 1)**2
            mean_se = values.std(ddof=1)/np.sqrt(len(values))
            mse_se = squared_error.std(ddof=1)/np.sqrt(len(values))
            expected_mean = .01 + p['bias']
            expected_var = p['single_block_variance']/t
            checks.append(dict(case_id=case_id, m=m, length=length, delta=delta,
                snr_db=20, blocks=t, groups=len(values), mean=float(values.mean()),
                mean_mc_se=float(mean_se), variance=float(values.var(ddof=1)),
                predicted_mean=expected_mean, predicted_variance=expected_var,
                normalized_mse=float(squared_error.mean()),
                normalized_mse_mc_se=float(mse_se),
                predicted_normalized_mse=p['normalized_mse'][str(t)],
                mean_z=float((values.mean()-expected_mean)/mean_se),
                mse_z=float((squared_error.mean()-p['normalized_mse'][str(t)])/mse_se)))
    np.savez_compressed(OUT/'group_estimates.npz', **arrays)
    report = dict(protocol='revisions/adversarial_20261113/operating_region_protocol.txt',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=1320261113, numpy=np.__version__, groups_per_mc_cell=20000,
        analytic=analytic, monte_carlo=checks,
        max_absolute_mean_z=max(abs(r['mean_z']) for r in checks),
        max_absolute_mse_z=max(abs(r['mse_z']) for r in checks),
        max_relative_variance_difference=max(abs(r['variance']/r['predicted_variance']-1) for r in checks))
    (OUT/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k,v in report.items() if k not in ['analytic','monte_carlo']}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
