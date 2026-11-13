"""Finite-CP waveform audit for ideal synchronized complex-baseband OFDM.

The continuous waveform of symbol b is
    s_b(t) = sum_k X[b,k] exp(2j*pi*q[k]*t/N) / sqrt(N), -CP <= t < N,
where q = N*fftfreq(N), time is in base-rate sample periods, and adjacent
symbols have independent data. Each symbol has its own rectangular CP-extended
window. The channel is a sum of ideal delayed paths with nonnegative delays;
gains are fixed throughout the supplied burst. This is a block-static channel
special case, with no Doppler, timing offset, CFO, TX/RX filters, or RF effects.

Oversampling zero-pads the SIGNED frequency bins and multiplies the unitary
IFFT by sqrt(OS). Thus samples every OS positions equal ifft(X,norm='ortho'),
and useful-symbol mean power equals mean(abs(X)**2), independent of OS. Fractional
path delays must lie exactly on the 1/OS grid: the channel applies ordinary
linear sample shifts to the entire finite burst, never circular shifts or a
per-symbol frequency response. A delay off this grid is rejected, not rounded.

When CP >= max(delays), every FFT input sample comes from the current symbol's
periodic extension, hence Y[k]=X[k]*sum_p g[p]*exp(-2j*pi*fftfreq(N)[k]*tau[p]).
CP < max(delays) is permitted for explicit failure experiments. Rectangular
symbol switching has spectral sidelobes: this is NOT a globally bandlimited
TX/RX chain. Analog filtering or pulse shaping adds effective memory and needs
its own CP/timing model. The tests validate this ideal waveform, not hardware
or a standardized system. The Nyquist bin is assigned to -1/2 cycles/sample.

Run: python waveform_model.py
"""
import json

import numpy as np


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be an integer')
    try:
        integer = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite integer') from exc
    if integer != value or integer < minimum or (maximum is not None and integer > maximum):
        raise ValueError(f'{name} must be an integer in the allowed range')
    return integer


def _delays(delays):
    delays = np.asarray(delays, dtype=float)
    if delays.ndim != 1 or not len(delays) or not np.all(np.isfinite(delays)) or np.any(delays < 0):
        raise ValueError('delays must be a nonempty finite nonnegative vector')
    return delays


def signed_frequencies(n_fft=64):
    """Frequency in cycles per BASE-RATE sample, in NumPy FFT array order."""
    return np.fft.fftfreq(_integer(n_fft, 'n_fft', 2))


def delay_atoms(delays, n_fft=64):
    """Return N x paths signed-baseband continuous-delay Fourier atoms."""
    return np.exp(-2j*np.pi*np.outer(signed_frequencies(n_fft), _delays(delays)))


def channel_response(delays, gains, n_fft=64):
    """Return (..., N) path response; gains have shape (..., paths)."""
    atoms = delay_atoms(delays, n_fft)
    gains = np.asarray(gains, dtype=complex)
    if gains.ndim < 1 or gains.shape[-1] != atoms.shape[1] or not np.all(np.isfinite(gains)):
        raise ValueError('gains must be finite with one last-axis value per path')
    return gains @ atoms.T


def modulate(symbols, cp_samples=16, oversample=20):
    """Generate finite concatenated CP symbols, shape (..., burst_samples).

    symbols has shape (..., n_symbols, N). CP is an integer number of base-rate
    samples between 0 and N. The first sample is the first symbol's CP start;
    there are no cyclic extensions outside the transmitted burst.
    """
    symbols = np.asarray(symbols, dtype=complex)
    if symbols.ndim < 2 or symbols.shape[-2] < 1 or not np.all(np.isfinite(symbols)):
        raise ValueError('symbols must be finite with shape (..., n_symbols, N)')
    n_fft = _integer(symbols.shape[-1], 'n_fft', 2)
    cp_samples = _integer(cp_samples, 'cp_samples', 0, n_fft)
    oversample = _integer(oversample, 'oversample', 1)
    length = n_fft*oversample
    padded = np.zeros(symbols.shape[:-1]+(length,), dtype=complex)
    bins = np.rint(signed_frequencies(n_fft)*n_fft).astype(int) % length
    padded[..., bins] = symbols
    waveform = np.fft.ifft(padded, axis=-1, norm='ortho')*np.sqrt(oversample)
    if cp_samples:
        waveform = np.concatenate([waveform[..., -cp_samples*oversample:], waveform], axis=-1)
    return waveform.reshape(symbols.shape[:-2]+(-1,))


def receive_ofdm(symbols, delays, gains, cp_samples=16, oversample=20):
    """Transmit, linearly delay/sum, sample, remove CP, then perform unitary FFT.

    symbols: (..., n_symbols, N); gains: (..., paths), broadcast batch axes.
    Output shape: (broadcast batch axes, n_symbols, N). Gains remain fixed over
    a burst; separate batch entries may have independent channels. Transmission
    is zero outside the finite burst. Noise is omitted to isolate the physical
    channel equivalence. Independent CN(0,v) noise on base-rate receiver samples
    remains CN(0,v I) after the unitary FFT; no oversampled noise PSD is claimed.
    """
    symbols = np.asarray(symbols, dtype=complex)
    transmitted = modulate(symbols, cp_samples, oversample)
    oversample = _integer(oversample, 'oversample', 1)
    cp_samples = int(cp_samples)
    delays = _delays(delays)
    scaled = delays*oversample
    if np.any(abs(scaled-np.rint(scaled)) > 1e-9):
        raise ValueError('delays must lie on the 1/oversample grid; increase oversample')
    shifts = np.rint(scaled).astype(int)
    gains = np.asarray(gains, dtype=complex)
    if gains.ndim < 1 or gains.shape[-1] != len(delays) or not np.all(np.isfinite(gains)):
        raise ValueError('gains must be finite with one last-axis value per path')
    batch_shape = np.broadcast_shapes(symbols.shape[:-2], gains.shape[:-1])
    transmitted = np.broadcast_to(transmitted, batch_shape+(transmitted.shape[-1],))
    gains = np.broadcast_to(gains, batch_shape+(len(delays),))
    burst_length = transmitted.shape[-1]
    received = np.zeros(batch_shape+(burst_length+int(shifts.max()),), dtype=complex)
    for path, shift in enumerate(shifts):
        received[..., shift:shift+burst_length] += transmitted*gains[..., path, None]
    n_symbols, n_fft = symbols.shape[-2:]
    indices = ((np.arange(n_symbols)[:, None]*(n_fft+cp_samples)+cp_samples
                + np.arange(n_fft))*oversample)
    return np.fft.fft(received[..., indices], axis=-1, norm='ortho')


def projection_noise_bias(delays, powers, pilots, length, n_fft=64):
    """E[RSS/(M-L)]-v under independent zero-mean path gains with given powers.

    Uses QR rather than assuming comb-pilot orthogonality. This is the standard
    projection trace identity, not a new result or a thermal-noise contribution.
    """
    delays = _delays(delays)
    powers = np.asarray(powers, dtype=float)
    if powers.shape != delays.shape or np.any(powers < 0) or not np.all(np.isfinite(powers)):
        raise ValueError('powers must be finite, nonnegative and match delays')
    raw_pilots = np.asarray(pilots)
    if raw_pilots.ndim != 1 or not np.all(np.isfinite(raw_pilots)):
        raise ValueError('pilots must be a finite integer vector')
    pilots = raw_pilots.astype(int)
    if np.any(pilots != raw_pilots) or len(np.unique(pilots)) != len(pilots) or np.any((pilots < 0) | (pilots >= n_fft)):
        raise ValueError('pilots must be distinct valid FFT indices')
    length = _integer(length, 'length', 1, len(pilots)-1)
    a = delay_atoms(np.arange(length), n_fft)[pilots]
    b = delay_atoms(delays, n_fft)[pilots]
    if np.linalg.matrix_rank(a) != length:
        raise ValueError('pilot basis must have full column rank')
    orthogonal, _ = np.linalg.qr(a, mode='reduced')
    residual_atoms = b-orthogonal@(orthogonal.conj().T@b)
    return float(np.sum(abs(residual_atoms)**2*powers)/(len(pilots)-length))


def audit_report(seed=1485):
    """Small reproducible physical-model check; does not rank estimators."""
    rng = np.random.default_rng(seed)
    symbols = (rng.standard_normal((8, 3, 64))+1j*rng.standard_normal((8, 3, 64)))/np.sqrt(2)
    powers = np.array([.5, .3, .2])
    gains = (rng.standard_normal((8, 3))+1j*rng.standard_normal((8, 3)))*np.sqrt(powers/2)
    errors = {}
    for name, delays in [('integer', [0, 7, 11]), ('fractional', [0, 7.35, 11.2])]:
        expected = symbols*channel_response(delays, gains)[:, None, :]
        actual = receive_ofdm(symbols, delays, gains, cp_samples=16, oversample=20)
        errors[name+'_max_abs_error'] = float(np.max(abs(actual-expected)))
    delays = [0, 7.35, 11.2]
    expected = symbols*channel_response(delays, gains)[:, None, :]
    short_cp = receive_ofdm(symbols, delays, gains, cp_samples=8, oversample=20)
    errors['short_cp_relative_rms_error'] = float(np.linalg.norm(short_cp[:, 1:]-expected[:, 1:])/np.linalg.norm(expected[:, 1:]))
    bias = {str(length): projection_noise_bias(delays, powers, np.arange(0, 64, 4), length) for length in [12, 15]}
    return {'status': 'ideal_rectangular_cp_ofdm_waveform_check', 'seed': seed,
            'numpy': np.__version__, 'n_fft': 64, 'oversample': 20,
            'sufficient_cp': 16, 'insufficient_cp': 8,
            'delays_in_base_sample_periods': delays, 'errors': errors,
            'fractional_residual_additive_bias_by_length': bias,
            'limits': 'Burst-static ideal paths; no TX/RX filtering, CFO, timing error, Doppler or RF impairments; exact grid delays only.'}


if __name__ == '__main__':
    print(json.dumps(audit_report(), indent=2))
