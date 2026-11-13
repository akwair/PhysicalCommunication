"""Generic CPU OMP plus variable-projection continuous-delay baseline.

This is an implementation of established sparse-estimation ingredients, not a
reproduction of NOMP or of a particular published algorithm. It processes one
complex pilot snapshot and a supplied scalar complex-noise variance E|w|^2.
Neither path count nor true delays, gains, profile, or SNR enter the estimator.

Candidate order k is selected by the predeclared known-noise BIC-type score
    2 * RSS / variance + d_k * log(2 * M),
where d_k = 3k for continuous delays and complex gains, or 2k for fixed-grid
complex gains. These different parameter counts mean refine=True/False is not
a fixed-order ablation. BIC is an asymptotic heuristic here: it ignores discrete
grid-search multiplicity, weak/unresolved path singularities, and uncertainty
in an externally estimated noise variance. The bounded local optimizer can
miss closely spaced paths; dense channels need not have a sparse description.

The candidate dictionary uses the supplied SIGNED frequencies directly:
    A[m, k] = exp(-2j * pi * frequency[m] * delay[k]).
There is no FFT padding, implicit wrapping of frequencies, or OFDM-size input.
"""

from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares


def _atoms(frequencies, delays):
    return np.exp(-2j * np.pi * np.outer(frequencies, delays))


@lru_cache(maxsize=32)
def _dictionary(frequencies, max_delay, grid_step):
    grid = np.append(np.arange(0.0, max_delay, grid_step), max_delay)
    atoms = _atoms(np.asarray(frequencies), grid)
    grid.flags.writeable = False
    atoms.flags.writeable = False
    return grid, atoms


def _refine(y, frequencies, initial_delays, max_delay):
    """Bounded variable projection with its full real-delay Jacobian."""
    cached_delays = None
    cached_values = None

    def evaluate(delays):
        nonlocal cached_delays, cached_values
        if cached_delays is None or not np.array_equal(cached_delays, delays):
            atoms = _atoms(frequencies, delays)
            inverse = np.linalg.pinv(atoms)
            gains = inverse @ y
            residual = atoms @ gains - y
            derivative = (-2j * np.pi * frequencies[:, None]) * atoms
            direct = derivative * gains[None, :]
            # Differentiating A^H r = 0 gives the second (residual) term.
            jacobian = direct - atoms @ (inverse @ direct)
            jacobian -= inverse.conj().T * (derivative.conj().T @ residual)[None, :]
            cached_delays = delays.copy()
            cached_values = (gains, residual, jacobian)
        return cached_values

    def objective(delays):
        residual = evaluate(delays)[1]
        return np.concatenate((residual.real, residual.imag))

    def jacobian(delays):
        derivative = evaluate(delays)[2]
        return np.vstack((derivative.real, derivative.imag))

    fitted = least_squares(
        objective, initial_delays, jac=jacobian, bounds=(0.0, max_delay),
        # Rectangular trust regions can reach an active delay bound exactly;
        # TRF's scaled gradient may stop with a small artificial boundary path.
        method="dogbox", max_nfev=60, ftol=1e-9, xtol=1e-9, gtol=1e-10,
    )
    gains, residual, _ = evaluate(fitted.x)
    return fitted.x, gains, residual, int(fitted.nfev), bool(fitted.success)


def estimate_offgrid(y, pilot_frequencies, all_frequencies, noise_variance,
                     max_delay=12.0, grid_step=0.25, max_paths=6, refine=True):
    """Estimate one frequency response without being supplied the path count.

    Frequencies are normalized cycles/sample; delays are in samples and are
    bounded to [0, max_delay]. The noise variance is per COMPLEX pilot sample.
    Supply known noise or a separately computed estimate unchanged across the
    comparison. max_paths caps the fitted order; it is not a true-order input.

    OMP adds the atom of greatest residual correlation, jointly refits complex
    gains, and optionally refines all delays by bounded variable projection.
    All visited orders including zero compete under the score above. Iteration
    stops early only when the next order's penalty alone exceeds the best
    score, making every subsequent order unable to win even at zero RSS.

    An effective variance floor of 100*eps*mean(abs(y)**2) avoids division by
    zero and unstable exact-fit selection in noiseless numerical checks. It
    has no calibrated statistical interpretation at zero supplied variance.
    The output exposes that floor, the effective variance, candidate scores,
    optimizer work, and elapsed CPU wall time for audit. nfev is the total
    nonlinear objective evaluations; elapsed_seconds includes dictionary setup
    on a cache miss. No parallelism or accelerator is used.
    """
    started = perf_counter()
    y = np.asarray(y, dtype=np.complex128)
    pilot_frequencies = np.asarray(pilot_frequencies, dtype=float)
    all_frequencies = np.asarray(all_frequencies, dtype=float)
    if y.ndim != 1 or y.size < 2 or pilot_frequencies.shape != y.shape:
        raise ValueError("y and pilot_frequencies must be equal-length 1D vectors (M >= 2)")
    if all_frequencies.ndim != 1 or all_frequencies.size == 0:
        raise ValueError("all_frequencies must be a nonempty 1D vector")
    if not all(np.all(np.isfinite(x)) for x in (y, pilot_frequencies, all_frequencies)):
        raise ValueError("observations and frequencies must be finite")
    noise_variance = float(noise_variance)
    max_delay, grid_step = float(max_delay), float(grid_step)
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    if not np.isfinite(max_delay) or max_delay <= 0:
        raise ValueError("max_delay must be finite and positive")
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("grid_step must be finite and positive")
    if isinstance(max_paths, bool) or not np.isfinite(max_paths) or int(max_paths) != max_paths or max_paths < 0:
        raise ValueError("max_paths must be a nonnegative integer")
    if not isinstance(refine, (bool, np.bool_)):
        raise ValueError("refine must be boolean")

    count = y.size
    parameters_per_path = 3 if refine else 2
    # Avoid fitting as many or more free real parameters than observations.
    order_cap = min(int(max_paths), (2 * count - 1) // parameters_per_path)
    grid, dictionary = _dictionary(tuple(pilot_frequencies), max_delay, grid_step)
    order_cap = min(order_cap, grid.size)
    energy = float(np.vdot(y, y).real)
    signal_scale = np.sqrt(energy / count) if energy > 0 else 1.0
    numerical_floor = max(100 * np.finfo(float).eps * energy / count,
                          np.finfo(float).tiny)
    effective_variance = max(noise_variance, numerical_floor)
    normalized_y = y / signal_scale
    residual = -normalized_y.copy()
    delays = np.empty(0, dtype=float)
    gains = np.empty(0, dtype=complex)
    penalty_step = parameters_per_path * np.log(2 * count)
    rss_history = [energy]
    score_history = [2 * energy / effective_variance]
    nfev_history = [0]
    refinement_success = [True]
    candidates = [(delays.copy(), gains.copy())]
    best_order = 0
    stopping_reason = "max_paths"

    for order in range(1, order_cap + 1):
        if order * penalty_step >= score_history[best_order]:
            stopping_reason = "penalty_lower_bound"
            break
        correlations = np.abs(dictionary.conj().T @ residual)
        if delays.size:
            used = np.min(np.abs(grid[:, None] - delays[None, :]), axis=1) < 1e-8
            correlations[used] = -np.inf
        if not np.any(np.isfinite(correlations)):
            stopping_reason = "dictionary_exhausted"
            break
        initial = np.append(delays, grid[int(np.argmax(correlations))])
        initial_atoms = _atoms(pilot_frequencies, initial)
        initial_gains = np.linalg.lstsq(initial_atoms, normalized_y, rcond=None)[0]
        initial_residual = initial_atoms @ initial_gains - normalized_y
        delays, gains, residual = initial, initial_gains, initial_residual
        evaluations, success = 0, True
        if refine:
            refined_delays, refined_gains, refined_residual, evaluations, success = _refine(
                normalized_y, pilot_frequencies, initial, max_delay)
            # A bounded optimizer should improve the seed, but retain the seed
            # if numerical roundoff or premature convergence makes it worse.
            if np.vdot(refined_residual, refined_residual).real <= np.vdot(residual, residual).real:
                delays, gains, residual = refined_delays, refined_gains, refined_residual
        permutation = np.argsort(delays)
        delays, gains = delays[permutation], gains[permutation]
        rss = float(np.vdot(residual, residual).real * signal_scale ** 2)
        score = 2 * rss / effective_variance + order * penalty_step
        rss_history.append(rss)
        score_history.append(float(score))
        nfev_history.append(evaluations)
        refinement_success.append(success)
        candidates.append((delays.copy(), gains.copy() * signal_scale))
        if score < score_history[best_order]:
            best_order = order

    selected_delays, selected_gains = candidates[best_order]
    fitted_response = _atoms(all_frequencies, selected_delays) @ selected_gains
    return {
        "response": fitted_response,
        "delays": selected_delays,
        "gains": selected_gains,
        "selected_order": best_order,
        "rss": rss_history[best_order],
        "selected_score": score_history[best_order],
        "score_history": score_history,
        "rss_history": rss_history,
        "nfev": sum(nfev_history),
        "nfev_history": nfev_history,
        "refinement_success": refinement_success,
        "noise_variance": noise_variance,
        "effective_noise_variance": effective_variance,
        "numerical_variance_floor": numerical_floor,
        "criterion": "known_noise_bic",
        "parameters_per_path": parameters_per_path,
        "order_cap": order_cap,
        "stopping_reason": stopping_reason,
        "elapsed_seconds": perf_counter() - started,
    }
