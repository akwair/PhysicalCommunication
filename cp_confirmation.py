"""Independent paired confirmation using a conventional CP noise statistic.

See confirmation_protocol.txt. Frequency-domain and CP-tail noise are coupled
through a unitary IFFT. No new estimation algorithm is proposed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pilot_ofdm import cn, N, PILOTS, DATA
from stage3_study import (T, L, M, FREQ, stage3_profiles, response_matrix,
                          projected_statistics, metrics)
from offgrid_baseline import estimate_offgrid
from waveform_model import modulate


def cp_noise_statistic(frequency_noise, prefix_noise, blocks=16):
    tail_noise = np.fft.ifft(frequency_noise[:, :blocks], axis=-1,
                             norm='ortho')[..., -4:]
    difference = prefix_noise[:, :blocks] - tail_noise
    return np.mean(np.abs(difference)**2 / 2, axis=(1, 2))


def generate_cp(rng, frames, delays, power, variance):
    gains = cn(rng, (frames, T, len(delays))) * np.sqrt(power)
    channels = gains @ response_matrix(delays).T
    frequency_noise = np.sqrt(variance) * cn(rng, (frames, T, N))
    prefix_noise = np.sqrt(variance) * cn(rng, (frames, T, 4))
    yp = channels[:, :, PILOTS] + frequency_noise[:, :, PILOTS]
    bits = rng.integers(0, 2, (frames, len(DATA), 2))
    symbols = ((1-2*bits[:, :, 0])+1j*(1-2*bits[:, :, 1]))/np.sqrt(2)
    yd = channels[:, 0, DATA]*symbols + frequency_noise[:, 0, DATA]
    return channels[:, 0], yp, bits, yd, cp_noise_statistic(frequency_noise, prefix_noise)


def cp_signal_audit(seed=113):
    """Independent waveform check of the four clean CP/tail signal pairs."""
    rng = np.random.default_rng(seed)
    x = cn(rng, (3, 64))
    tx = modulate(x, cp_samples=16, oversample=20)
    errors = {}
    for name, (delays, power) in stage3_profiles().items():
        gains = cn(rng, (len(delays),)) * np.sqrt(power)
        shifts = np.rint(delays * 20).astype(int)
        rx = np.zeros(len(tx)+max(shifts), complex)
        for shift, gain in zip(shifts, gains):
            rx[shift:shift+len(tx)] += gain * tx
        # Symbol 1: CP indices12..15 vs useful indices60..63, 64 apart.
        left = (80 + np.arange(12,16))*20
        errors[name] = float(np.max(abs(rx[left]-rx[left+64*20])))
    return errors


def run(args):
    rng = np.random.default_rng(args.seed)
    raw, rows, diagnostics = {}, [], []
    for profile in ['fractional_sparse', 'close_fractional']:
        delays, power = stage3_profiles()[profile]
        for snr in [10,20]:
            v = 10**(-snr/10)
            truth, yp, bits, yd, cp = generate_cp(rng,args.frames,delays,power,v)
            z, vr, basis = projected_statistics(yp)
            sources = {'residual16': vr.mean(axis=1), 'cp16': cp,
                       'known_noise': np.full(args.frames,v)}
            for source, vv in sources.items():
                raw[f'{profile}_{snr}_{source}_noise'] = vv
                if source != 'known_noise':
                    estimate = (z[:,0]*np.maximum(0, 1-vv[:,None]/M/
                                  np.maximum(abs(z[:,0])**2,1e-300))) @ basis.T
                    raw[f'{profile}_{snr}_eb_{source}'] = metrics(estimate,truth,bits,yd)
                estimates, order, calls = [], [], []
                for i in range(args.frames):
                    fit = estimate_offgrid(yp[i,0],FREQ[PILOTS],FREQ,
                          max(vv[i],1e-12),max_delay=12,grid_step=.25,
                          max_paths=6,refine=True)
                    estimates.append(fit['response'])
                    order.append(fit['selected_order'])
                    # Store complete diagnostics without channel truth/received samples.
                    calls.append({k: val for k,val in fit.items()
                                  if k not in ('response','delays','gains')})
                name = f'refined_{source}'
                raw[f'{profile}_{snr}_{name}'] = metrics(np.asarray(estimates),truth,bits,yd)
                raw[f'{profile}_{snr}_{name}_orders'] = np.asarray(order)
                diagnostics.append(dict(profile=profile,snr_db=snr,source=source,calls=calls))
            for name in ['eb_residual16','eb_cp16','refined_residual16','refined_cp16','refined_known_noise']:
                values = raw[f'{profile}_{snr}_{name}']
                rows.append(dict(profile=profile,snr_db=snr,method=name,
                    nmse_db=float(10*np.log10(values[0].sum()/values[1].sum())),
                    ber=float(values[2].sum()/(96*args.frames))))
            print(f'{profile} {snr} dB complete',flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'frame_metrics.npz',**raw)
    def encode(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,complex): return [value.real,value.imag]
        raise TypeError(type(value).__name__)
    record = dict(seed=args.seed,groups_per_cell=args.frames,results=rows,
        signal_cancellation_audit=cp_signal_audit(),diagnostics=diagnostics,
        source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['cp_confirmation.py','offgrid_baseline.py','stage3_study.py','confirmation_protocol.txt']})
    (args.out/'results.json').write_text(json.dumps(record,default=encode,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,default=300)
    parser.add_argument('--seed',type=int,default=1120261113)
    parser.add_argument('--out',type=Path,default=Path('results/cp_confirmation1'))
    run(parser.parse_args())
