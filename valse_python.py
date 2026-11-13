"""CPU implementation of the VALSE paper equations with Heuristic 2.

Badiu, Hansen and Fleury, IEEE TSP 65(9), 2247--2261 (2017),
https://arxiv.org/html/1604.03744 . The author demonstration was inspected;
numerical equivalence to its MATLAB execution has NOT been established.
See valse_implementation_notes.txt for explicit differences and validation.
No true signal, true order, or true noise variance enters the estimator.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz
from scipy.special import ive

KAPPA_MAX = 1e12
VARIANCE_FLOOR = 1e-12  # relative to max(abs(input))**2


def _count(stats, name, number=1):
    if stats is not None and number:
        stats[name] = stats.get(name, 0) + int(number)


def Ainv(R, *, stats=None):
    """Mardia--Jupp approximation to inverse I1(kappa)/I0(kappa).

    Exact R=1 is regularized to finite KAPPA_MAX, recorded in stats.
    """
    r = np.asarray(R, dtype=float)
    if np.any(~np.isfinite(r)) or np.any((r < 0) | (r > 1)):
        raise ValueError('R must lie in [0, 1]')
    ceiling = 1 - .5 / KAPPA_MAX
    _count(stats, 'resultant_clips', np.count_nonzero(r > ceiling))
    r = np.minimum(r, ceiling)
    out = np.empty_like(r)
    first, last = r < .53, r >= .85
    middle = ~(first | last)
    t = r[first]**2
    out[first] = r[first] * (2 + t + 5*t*t/6)
    out[middle] = -.4 + 1.39*r[middle] + .43/(1-r[middle])
    out[last] = 1/(r[last]*(r[last]-1)*(r[last]-3))
    return np.minimum(out, KAPPA_MAX)


def bessel_ratio(order, kappa, *, stats=None):
    """I_|n|(k)/I0(k), integer n; stable at uniform and concentrated limits."""
    n, k = np.broadcast_arrays(np.abs(np.asarray(order, dtype=float)),
                               np.asarray(kappa, dtype=float))
    if np.any(~np.isfinite(k)) or np.any(k < 0) or np.any(n != np.floor(n)):
        raise ValueError('Require finite kappa >= 0 and integer orders')
    result = np.empty_like(k)
    large = k > 1e6
    if np.any(large):
        # Leading high-concentration expansion, including the 1/k**2 term.
        # This avoids AMOS/ive loss of finiteness at enormous kappa.
        result[large] = np.exp(-n[large]**2/(2*k[large])
                               - n[large]**2/(4*k[large]**2))
        _count(stats, 'large_kappa_asymptotic', np.count_nonzero(large))
    regular = ~large
    if np.any(regular):
        result[regular] = ive(n[regular], k[regular]) / ive(0, k[regular])
    result[n == 0] = 1.
    return np.clip(result, 0, 1)


def von_mises_moments(n, mu, kappa, *, stats=None):
    """Integer moments E[exp(j*n*Theta)] of VM(mu,kappa), not point atoms."""
    n = np.asarray(n)
    return np.exp(1j*n*mu) * bessel_ratio(n, kappa, stats=stats)


def _round_away(x):
    return np.copysign(np.floor(np.abs(x)+.5), x)


def heuristic2(eta, m, *, full_length=None, stats=None):
    """Author Heuristic 2: wrapped-VM branches and one Newton refinement.

    The zero-order factor is constant and is excluded before branch building.
    A completely flat likelihood returns a uniform distribution.
    """
    eta = np.asarray(eta, dtype=complex).reshape(-1)
    m = np.asarray(m, dtype=int).reshape(-1)
    if len(m) != len(eta):
        raise ValueError('eta and m must have the same length')
    if full_length is None:
        full_length = int(m.max())+1 if len(m) else 1
    use = (m > 0) & (np.abs(eta) > 0)
    indices, values = m[use], eta[use]
    if not len(indices):
        return von_mises_moments(np.arange(full_length),0.,0.), 0., 0.
    kappas = np.abs(values)
    resultants = bessel_ratio(1, kappas, stats=stats)
    kmix = Ainv(resultants**(1/indices.astype(float)**2), stats=stats)
    last = indices[-1]
    q = kmix[-1] * np.exp(1j*(np.angle(values[-1])+2*np.pi*np.arange(1,last+1))/last)
    for index, value, strength in zip(indices[-2::-1], values[-2::-1], kmix[-2::-1]):
        phi = np.angle(value)
        turns = _round_away((index*np.angle(q)-phi)/(2*np.pi))
        q += strength * np.exp(1j*(phi+2*np.pi*turns)/index)
    winner = int(np.argmax(np.abs(q)))
    mu = np.angle(q[winner])
    exponential = np.exp(1j*indices*mu)
    d1 = -np.imag(np.vdot(values, indices*exponential))
    d2 = -np.real(np.vdot(values, indices**2*exponential))
    if d2 < 0:
        theta = mu - d1/d2
        kappa = float(Ainv(np.exp(.5/d2), stats=stats))
    else:
        theta = mu
        kappa = float(min(abs(q[winner]),KAPPA_MAX))
        _count(stats,'nonconcave_frequency_updates')
    theta = float((theta+np.pi)%(2*np.pi)-np.pi)
    return von_mises_moments(np.arange(full_length),theta,kappa,stats=stats), theta, kappa


def _positive(value, floor, stats, name):
    if not np.isfinite(value):
        raise FloatingPointError(f'Nonfinite {name}')
    if value < floor:
        _count(stats,name)
        return floor
    return float(value)


def _support_objective(J,h,s,nu,rho,tau):
    ids = np.flatnonzero(s)
    if not len(ids):
        return 0.
    precision = J[np.ix_(ids,ids)]/nu + np.eye(len(ids))/tau
    sign, logdet = np.linalg.slogdet(precision)
    if sign.real <= 0:
        raise FloatingPointError('Support precision is not positive definite')
    v = np.linalg.solve(precision,h[ids]/nu)
    return float(-logdet + np.vdot(h[ids]/nu,v).real
                 + len(ids)*np.log(rho/(1-rho)/tau))


def maxZ(J,h,M,nu,rho,tau,*,return_trace=False,stats=None):
    """Empty-start greedy support search, Appendix A, with rank-one updates.

    Returns (K, boolean support, full gain means, full covariance). Inactive
    entries retain source-style scratch values; only supported entries define
    the posterior. return_trace additionally returns direct-check snapshots.
    The author demonstration restriction K < M-1 is retained for L > 1.
    """
    J = np.asarray(J,dtype=complex)
    h = np.asarray(h,dtype=complex).reshape(-1)
    L = len(h)
    if nu <= 0 or tau <= 0 or not 0 < rho < 1:
        raise ValueError('Require positive nu,tau and 0<rho<1')
    s = np.zeros(L,dtype=bool)
    w = np.zeros(L,dtype=complex)
    C = np.zeros((L,L),dtype=complex)
    trace = []
    cnst = np.log(rho/(1-rho)/tau)
    for _ in range(max(100,100*L)):
        ids, outside = np.flatnonzero(s),np.flatnonzero(~s)
        K = len(ids)
        delta = np.full(L,-np.inf)
        u = np.zeros(L,dtype=complex)
        v = np.zeros(L)
        if (K < M-1 or L == 1) and len(outside):
            cross = J[np.ix_(ids,outside)]
            cov_cross = C[np.ix_(ids,ids)] @ cross
            denominator = M+nu/tau-np.sum(cross.conj()*cov_cross,axis=0).real/nu
            floor = np.finfo(float).eps*max(1,M+nu/tau)*32
            _count(stats,'schur_complement_clips',np.count_nonzero(denominator<floor))
            denominator = np.maximum(denominator,floor)
            v[outside] = nu/denominator
            u[outside] = v[outside]*(h[outside]-cross.conj().T@w[ids])/nu
            delta[outside] = np.log(v[outside])+np.abs(u[outside])**2/v[outside]+cnst
        if K:
            diagonal = C[ids,ids].real
            delta[ids] = -np.log(diagonal)-np.abs(w[ids])**2/diagonal-cnst
        k = int(np.argmax(delta))
        if delta[k] <= 1e-12:
            result = (int(s.sum()),s,w,C)
            return result+(trace,) if return_trace else result
        if not s[k]:
            q = C[np.ix_(ids,ids)] @ J[ids,k]/nu
            w[k] = u[k]
            w[ids] -= q*u[k]
            C[np.ix_(ids,ids)] += v[k]*np.outer(q,q.conj())
            C[ids,k] = -v[k]*q
            C[k,ids] = C[ids,k].conj()
            C[k,k] = v[k]
            s[k] = True
            action = 'activate'
        else:
            s[k] = False
            ids = np.flatnonzero(s)
            w[ids] -= C[ids,k]*w[k]/C[k,k]
            C[np.ix_(ids,ids)] -= np.outer(C[ids,k],C[k,ids])/C[k,k]
            action = 'deactivate'
        C = (C+C.conj().T)/2
        if return_trace:
            ids = np.flatnonzero(s)
            trace.append(dict(support=ids.copy(),mean=w[ids].copy(),
                              covariance=C[np.ix_(ids,ids)].copy(),action=action,
                              objective=_support_objective(J,h,s,nu,rho,tau)))
    raise RuntimeError('Support search exceeded its numerical cycling guard')


def noise_variance_update(y, A, w, C):
    """Equation (24), evaluated without subtracting large squared norms."""
    M = len(y)
    J = A.conj().T@A
    np.fill_diagonal(J,M)
    uncertainty = np.maximum(0,1-np.sum(np.abs(A)**2,axis=0)/M)
    return float(np.linalg.norm(y-A@w)**2/M + np.trace(J@C).real/M
                 + np.sum(np.abs(w)**2*uncertainty))


def frequency_update_vector(y,A,w,C,support,component,nu,*,covariance_column=None):
    """Equation (17) with full candidate indexing for gain cross-covariances.

    A contains observed rows only. covariance_column overrides the globally
    indexed column solely to inspect the author demonstration's index choice.
    """
    ids = np.flatnonzero(support)
    others = ids[ids != component]
    column = component if covariance_column is None else covariance_column
    residual = y-A[:,others]@w[others]
    return 2/nu*(residual*np.conj(w[component])-A[:,others]@C[others,column])


def estimate_valse(y,m=None,max_iter=500,heuristic=2,*,covariance_index='equation',
                   variance_floor=VARIANCE_FLOOR):
    """Estimate one complex line-spectrum vector without oracle parameters.

    m: strictly increasing, nonnegative integer sample indices; defaults 0:M-1.
    max_iter counts outer updates (MATLAB additionally counts initialization).
    heuristic=2 retains VM frequency uncertainty. Other heuristics unsupported.
    covariance_index='equation' uses C[other,global_component] in Eq. (17).
    'source' uses the author-code local-column index for sensitivity studies;
    it does not imply numerical equivalence to an executed MATLAB reference.

    Full moments/reconstruction refer only to integer indices 0:max(m).
    No noninteger interpolation or physical delay-branch mapping is implied.
    """
    y = np.asarray(y,dtype=complex).reshape(-1)
    M = len(y)
    if not M or np.any(~np.isfinite(y)):
        raise ValueError('y must be a finite nonempty vector')
    raw_m = np.arange(M) if m is None else np.asarray(m)
    if (raw_m.ndim != 1 or len(raw_m)!=M or np.any(~np.isfinite(raw_m))
        or np.any(raw_m<0) or np.any(raw_m!=np.floor(raw_m))
        or np.any(np.diff(raw_m)<=0)):
        raise ValueError('m must be strictly increasing nonnegative integers')
    m = raw_m.astype(int)
    if heuristic != 2:
        raise ValueError('This implementation supports Heuristic 2 only')
    if covariance_index not in ('equation','source'):
        raise ValueError("covariance_index must be 'equation' or 'source'")
    if not isinstance(max_iter,(int,np.integer)) or max_iter<1:
        raise ValueError('max_iter must be a positive integer')
    if not np.isfinite(variance_floor) or variance_floor<=0:
        raise ValueError('variance_floor must be finite and positive')
    N = L = int(m[-1])+1
    scale = float(np.max(np.abs(y)))
    stats = {}
    if scale == 0:
        return dict(y_estimate=np.zeros(M,complex),x_estimate=np.zeros(N,complex),
                    moments=np.empty((N,0),complex),freqs=np.empty(0),amps=np.empty(0,complex),
                    gain_covariance=np.empty((0,0),complex),kappa=np.empty(0),noise_var=0.,
                    gain_prior_var=0.,iterations=0,converged=True,support=np.zeros(L,bool),
                    support_history=np.array([0]),stability=stats,covariance_index=covariance_index)
    yn = y/scale
    y2 = np.vdot(yn,yn).real
    A = np.zeros((N,L),complex)
    J = np.zeros((L,L),complex)
    h = np.zeros(L,complex)
    w = np.zeros(L,complex)
    C = np.zeros((L,L),complex)
    frequencies,kappas = np.zeros(L),np.zeros(L)
    residual = yn.copy()
    initial_K = max(1,L//2)
    rho = initial_K/L if L>1 else .5
    old_x = np.zeros(N,complex)
    for l in range(L):
        embedded = np.zeros(N,complex)
        embedded[m] = residual
        # np.correlate gives sum_k embedded[k+lag]*conj(embedded[k]).
        corr = np.correlate(embedded,embedded,mode='full')[N-1:]
        if l == 0:
            eigenvalues = np.linalg.eigvalsh(toeplitz(corr)/N)
            nu = _positive(float(np.mean(eigenvalues[:max(1,N//4)])),
                           variance_floor,stats,'noise_floor_clips')
            tau = _positive((y2/M-nu)/(rho*L),variance_floor,stats,'prior_floor_clips')
        eta = 2*corr[1:]/(M+nu/tau)/nu
        _,frequencies[l],kappas[l] = heuristic2(eta,np.arange(1,N),
                                               full_length=N,stats=stats)
        # Initialization fills observed rows only, matching the demo schedule.
        A[m,l] = von_mises_moments(m,frequencies[l],kappas[l],stats=stats)
        J[:l,l] = A[m,:l].conj().T@A[m,l]
        J[l,:l] = J[:l,l].conj()
        J[l,l] = M
        h[l] = np.vdot(A[m,l],yn)
        q = C[:l,:l]@J[:l,l]/nu
        denominator = M+nu/tau-np.vdot(J[:l,l],q).real
        denominator = _positive(denominator,np.finfo(float).eps*max(1,M+nu/tau)*32,
                                stats,'schur_complement_clips')
        v = nu/denominator
        u = v*(h[l]-np.vdot(J[:l,l],w[:l]))/nu
        w[l] = u
        w[:l] -= q*u
        C[:l,:l] += v*np.outer(q,q.conj())
        C[:l,l] = -v*q
        C[l,:l] = C[:l,l].conj()
        C[l,l] = v
        residual = yn-A[m,:l+1]@w[:l+1]
        if l+1 == initial_K:
            old_x = A[:,:l+1]@w[:l+1]
    history = [initial_K]
    converged = False
    for iteration in range(1,max_iter+1):
        K,s,w,C = maxZ(J,h,M,nu,rho,tau,stats=stats)
        ids = np.flatnonzero(s)
        if K:
            nu = _positive(noise_variance_update(yn,A[m][:,ids],w[ids],C[np.ix_(ids,ids)]),
                           variance_floor,stats,'noise_floor_clips')
            tau = _positive((np.vdot(w[ids],w[ids]).real+np.trace(C[np.ix_(ids,ids)]).real)/K,
                            variance_floor,stats,'prior_floor_clips')
            rho = min(K,L-1)/L if L>1 else .5
        else:
            rho = 1/L if L>1 else .5
        for local_index,k in enumerate(ids):
            column = k if covariance_index=='equation' else local_index
            eta = frequency_update_vector(yn,A[m],w,C,s,k,nu,covariance_column=column)
            A[:,k],frequencies[k],kappas[k] = heuristic2(eta,m,full_length=N,stats=stats)
        if K:
            J[:,ids] = A[m].conj().T@A[m][:,ids]
            J[ids,:] = J[:,ids].conj().T
            np.fill_diagonal(J,M)
            h[ids] = A[m][:,ids].conj().T@yn
        xr = A[:,ids]@w[ids]
        difference = np.linalg.norm(xr-old_x)
        old_norm = np.linalg.norm(old_x)
        history.append(K)
        if (old_norm>0 and difference/old_norm<1e-6) or (old_norm==0 and difference==0):
            converged = True
            break
        old_x = xr
    return dict(y_estimate=xr[m]*scale,x_estimate=xr*scale,moments=A[:,ids].copy(),
                freqs=frequencies[ids].copy(),amps=w[ids]*scale,
                gain_covariance=C[np.ix_(ids,ids)]*scale*scale,kappa=kappas[ids].copy(),
                noise_var=nu*scale*scale,gain_prior_var=tau*scale*scale,
                iterations=iteration,converged=converged,support=s.copy(),
                support_history=np.asarray(history),stability=stats,covariance_index=covariance_index)
