"""Correlated-channel audit of residual-noise pooling (standard Gaussian moments).

Not a new estimator or theorem. Checks how optimistic independent-block pooling
is when channel gains remain correlated over a calibration window.
"""
import json
from pathlib import Path
import numpy as np
from stage3_study import response_matrix, projected_statistics, M, L, PILOTS
from pilot_ofdm import cn


def predicted_moments(delays, power, variance, rho, blocks):
    a = response_matrix(np.arange(L))[PILOTS]
    b = response_matrix(delays)[PILOTS]
    q = np.eye(M)-a @ a.conj().T/M
    signal = q @ ((b*power) @ b.conj().T) @ q
    total = signal+variance*q
    dof = M-L
    mean = np.trace(total).real/dof
    cross_sum = sum((blocks-lag)*rho**(2*lag) for lag in range(1,blocks))
    var = (blocks*np.trace(total @ total).real+2*np.trace(signal @ signal).real*cross_sum)/(blocks*blocks*dof*dof)
    return float(mean), float(var)


def run(groups=20000, seed=920261113):
    rng = np.random.default_rng(seed)
    delays, power, variance = np.array([0,7.35,11.2]), np.array([.5,.3,.2]), .01
    b = response_matrix(delays)[PILOTS]
    rows = []
    for rho in [0,.9,.99,1.]:
        gains = np.empty((groups,16,3),dtype=complex)
        gains[:,0] = cn(rng,(groups,3))*np.sqrt(power)
        for block in range(1,16):
            gains[:,block] = rho*gains[:,block-1]+np.sqrt(1-rho*rho)*cn(rng,(groups,3))*np.sqrt(power)
        y = gains @ b.T+np.sqrt(variance)*cn(rng,(groups,16,M))
        _, vhat, _ = projected_statistics(y)
        for blocks in [1,4,16]:
            pooled = vhat[:,:blocks].mean(axis=1)
            mean, var = predicted_moments(delays,power,variance,rho,blocks)
            assert abs(pooled.mean()/mean-1)<.04
            assert abs(pooled.var()/var-1)<.10
            rows.append(dict(rho=rho,blocks=blocks,mean_ratio=float(pooled.mean()/variance),
                             relative_std=float(pooled.std()/variance),
                             predicted_mean_ratio=mean/variance,predicted_relative_std=np.sqrt(var)/variance))
    result = dict(status='gaussian_moment_audit_not_novel_theorem',groups=groups,seed=seed,results=rows)
    Path('results/stage3_correlation_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    run()
